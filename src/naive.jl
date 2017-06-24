using MLBase
using MLKernels
using DataFrames

__precompile__()

@inline logsig(x) = 1.0/(1.0+exp(-x))

type RVMSpec
    kernel::MLKernels.Kernel
    n_iter::Integer
    normalizer::Function
    threshold::Real
    tol::Real
end

type RVMFit{R <: Real}
    kernel::MLKernels.Kernel
    normal::Function
    labelmap::MLBase.LabelMap
    converged::Bool
    steps::Integer
    w::Vector{R}
    RV::Matrix{R}
end

function standard_score(Obs::AbstractMatrix)
    μ = mean(Obs, 1)
    σ = var(Obs, 1)

    f(M) = (M .- μ) ./ σ
end

function identitize(Obs::AbstractMatrix)
    identity
end

function get_data(ifname::AbstractString, target::Symbol)
    data = DataFrames.readtable(ifname)
    split_target(data, target)
end

function split_target(data::DataFrame, target::Symbol)
    Obs  = convert(Matrix{Float64},
                   data[setdiff(names(data), [target])])
    ObsT = convert(Vector, data[target])
    Obs, ObsT
end

function _predicted_probability{R <: Real}(Model::RVMFit, Features::AbstractMatrix{R})
    X = Model.normal(Features)
    _ = MLKernels.kernelmatrix(Model.kernel, X, Model.RV)
    ϕ = [_ ones(size(_, 1))] # augment with bias

    P = _classify(Model.w, ϕ)
    P
end

function predict{R <: Real}(Model::RVMFit, Values::AbstractMatrix{R})
    _ = convert(Vector{Int64}, (_predicted_probability(Model, Values) .>= 0.5))
    Y = _ .+ 1 # booleans map to {0, 1} , {en,de}coders map to {1, 2}
    labeldecode(Model.labelmap, Y)
end

function fit{R <: Real}(S::RVMSpec, Obs::AbstractMatrix{R}, ObsT::AbstractVector)
    # sample count, features
    n, p = size(Obs)

    # standardize normalizing process
    normalize = S.normalizer(Obs)
    X = normalize(Obs)

    # Relevance Vectors
    RV = X

    # standardize encoding process
    lm = labelmap(ObsT)
    t  = labelencode(lm, ObsT) - 1

    # design matrix / basis functions
    _ = MLKernels.kernelmatrix(S.kernel, X, X)
    ϕ = [_ ones(size(_, 1))] # augment with bias
    basis_count = size(ϕ, 2)

    # initialize priors N(0, 1)
    μ  = zeros(basis_count)
    α₀ = ones(basis_count) .* 1e0
    α₁ = α₀[1:end]

    # reporting information
    steps = S.n_iter
    succ  = false

    for i in 1:S.n_iter

        # update values
        μ, Σ = _solve_post_prob(μ, α₁, ϕ, t)
        γ  = 1 - (α₁ .* diag(Σ))
        α₁ = γ ./ (μ .^ 2)

        # identify informative vectors
        keep::Vector{Bool} = α₁ .< S.threshold
        if !any(keep)
            keep[1] = true # require at least one vecor
        end
        keep[end] = true # save bias

        # downselect uninformative vectors and weights
        α₀ = α₀[keep]
        α₁ = α₁[keep]
        γ  = γ[keep]
        ϕ  = ϕ[:, keep]
        Σ  = Σ[keep, keep]
        μ  = μ[keep]
        RV = RV[keep[1:end-1], :]

        # check for brreak
        Δα = maximum(abs(α₁ .- α₀))
        if (Δα < S.tol) & (steps > 2)
            steps = i
            succ  = true
            break
        end
        α₀ = α₁[1:end]
    end

    # Trained model as object
    modelfit::RVMFit{R} =
        RVMFit(S.kernel, normalize, lm, succ, steps, μ, RV)
end

function _classify(μ::AbstractVector, ϕ::AbstractMatrix)
    #@show "_classify"
    _ = ϕ * μ
    y = [logsig(x) for x in _]
    y
end

function _log_solve_post_prob(w::AbstractVector, α::AbstractVector,
                              ϕ::AbstractMatrix, t::AbstractVector)
    #@show "_log_post_probability"
    A = diagm(α)
    y = _classify(w, ϕ)
    pos = y[t .== 1]
    neg = y[t .!= 1]

    log_p = (0.5 * w' * A * w)[1] - sum(log(pos)) - sum(log(1 - neg))
end

function _hessian(w::AbstractVector, α::AbstractVector,
                  ϕ::AbstractMatrix, t::AbstractVector)
    #@show "_hessian"
    A = diagm(α)
    y = _classify(w, ϕ)
    B = diagm(y .* (1 - y))

    H = (ϕ' * B * ϕ + A)
    H
end

function _gradient(w::AbstractVector, α::AbstractVector,
                   ϕ::AbstractMatrix, t::AbstractVector)
    #@show "_gradient"
    A = diagm(α)
    y = _classify(w, ϕ)
    G = ϕ' * (t .- y) - (A * w)
    -G
end

function _newton_method(X₀::AbstractVector, F::Function, # Initial, Function
                        ∇::Function, ∇∇::Function)       # Gradient, Hessian
    #@show "_newton_method"
    while true
        #@show F(X₀) # uncomment to watch converge
        X₁ = X₀ - inv(∇∇(X₀)) * ∇(X₀)

        # convergence
        ΔX = maximum(abs(X₁ .- X₀))
        X₀ = X₁[1:end]
        if ΔX < 1e-3
            break
        end
    end
    X₀
end

function _solve_post_prob(w::AbstractVector, α::AbstractVector,
                           ϕ::AbstractMatrix, t::AbstractVector)
    w = _newton_method(w,
                       (w) -> _log_solve_post_prob(w, α, ϕ, t),
                       (w) -> _gradient(w, α, ϕ, t),
                       (w) -> _hessian(w, α, ϕ, t),
                       )

    Σ = inv(_hessian(w, α, ϕ, t))
    w, Σ
end

SIZE = 100, 1000 , 2300, 4300
Train, TrainT = get_data("../data/training.csv", :class)

for K in (MLKernels.LinearKernel(), MLKernels.RadialBasisKernel(), MLKernels.GaussianKernel())
    @show "******************"
    @show K
    spec  = RVMSpec(K, 50, standard_score, 1e5, 1e-6)
    for S in SIZE
        @show S
        @time model = fit(spec, Train[1:S, :], TrainT[1:S])
        @show size(model.RV)
        TrainP = predict(model, Train)
        @show confusmat(2,
                        labelencode(model.labelmap, TrainP),
                        labelencode(model.labelmap, TrainT))
        @show mean(TrainP .== TrainT)

        Test, TestT = get_data("../data/testing.csv", :class)
        TestP = predict(model, Test)
        @show confusmat(2,
                        labelencode(model.labelmap, TestP),
                        labelencode(model.labelmap, TestT))
        @show mean(TestP .== TestT)
    end
end
