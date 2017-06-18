
using MLBase
using MLKernels
using DataFrames
#using Convex

__precompile__()

@inline expit(x) = 1/(1+exp(-x))

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
    encode::Function
    converged::Bool
    steps::Integer
    w::Vector{R}
    RV::Matrix{R}
end

function whitening(Obs::AbstractMatrix)
    μ = mean(Obs, 1)
    σ = var(Obs, 1)

    f(M) = (M .- μ) ./ σ
end

function identitize(Obs::AbstractMatrix)
    identity
end

function get_data(ifname::AbstractString, target::Symbol)
    data   = DataFrames.readtable(ifname)

    Obs  = convert(Matrix{Float64},
                   data[setdiff(names(data), [target])])
    ObsT = convert(Vector, data[target])
    Obs, ObsT
end

function predict{R <: Real}(Model::RVMFit, Values::AbstractMatrix{R})
    X = Model.normal(Values)
    _ = MLKernels.kernelmatrix(Model.kernel, X, Model.RV)
    ϕ = [_ ones(size(_, 1))] # augment with bias

    P = _classify(Model.w, ϕ)
end

function fit{R <: Real}(S::RVMSpec, Obs::AbstractMatrix{R}, ObsT::AbstractVector)
    # sample count, features
    n, p = size(Obs)

    # standardize normalizing process
    normalize = S.normalizer(Obs)
    X  = normalize(Obs)

    # Relevance Vectors
    RV = X

    # standardize encoding process
    lm = labelmap(ObsT)
    encode(V) = labelencode(lm, V) .- 1
    @show t  = encode(ObsT)

    # design matrix / basis functions
    _ = MLKernels.kernelmatrix(S.kernel, X, X)
    ϕ = [_ ones(size(_, 1))] # augment with bias
    basis_count = size(ϕ, 2)

    # initialize priors
    α₀ = ones(basis_count) .* 1e-6
    α₁ = α₀[1:end]
    μ  = zeros(basis_count)

    # reporting information
    steps = S.n_iter
    succ  = false

    for i in 1:S.n_iter

        # update values
        μ, Σ = _posterior(μ, α₀, ϕ, t)
        γ  = 1 .- (α₀ .* diag(Σ))
        α₁ = γ ./ (μ .^ 2)

        # identify informative vectors
        keep = α₁ .< S.threshold
        if !any(keep)
            keep[1] = true
        end
        keep[end] = true # save bias

        @show sum(keep)

        # downselect uninformative vectors
        α₀ = α₀[keep]
        α₁ = α₁[keep]
        γ  = γ[keep]
        ϕ  = ϕ[:, keep]
        Σ  = Σ[keep, keep]
        μ  = μ[keep]
        RV = RV[keep[1:end-1], :]

        # check for brreak
        @show Δα = maximum(abs(α₁ .- α₀))
        if Δα < S.tol
            steps = i
            succ  = true
            break
        end
        α₀ = α₁[1:end]
    end

    # Trained model as object
    modelfit::RVMFit{R} =
        RVMFit(S.kernel, normalize, encode, succ, steps, μ, RV)
end

function _classify(μ::AbstractVector, ϕ::AbstractMatrix)
    @show "_classify"
    y = [expit(x) for x in ϕ * μ]
    y
end

function _log_posterior(w::AbstractVector, α::AbstractVector,
                        ϕ::AbstractMatrix, t::AbstractVector)
    @show "_log_posterior"
    A = diagm(α)
    y = _classify(w, ϕ)
    pos = y[t .== 1]
    neg = y[t .!= 1]
    @show size(pos)
    @show size(neg)

    log_p = (0.5 * w' * A * w)[1] - sum(log(pos)) - sum(log(1 - neg))
end

function _hessian(w::AbstractVector, α::AbstractVector,
                  ϕ::AbstractMatrix, t::AbstractVector)
    @show "_hessian"
    A = diagm(α)
    y = _classify(w, ϕ)
    B = diagm(y .* (1 - y))

    H = - (ϕ' * B * ϕ + A)
    H
end

function _gradient(w::AbstractVector, α::AbstractVector,
                   ϕ::AbstractMatrix, t::AbstractVector)
    @show "_gradient"
    A = diagm(α)
    y = _classify(w, ϕ)
    G = ϕ' * (t .- y) - (A * w)
    G
end

function _newton_method(X₀::AbstractVector, ∇∇::Function, ∇::Function, F::Function)
    @show "_newton_method"
    while true
        @show F(X₀)

        X₁ = X₀ - (inv(∇∇(X₀)) * ∇(X₀))

        # convergence
        ΔX = maximum(abs(X₁ .- X₀))
        X₀ = X₁[1:end]
        if ΔX < 1e-3
            break
        end
    end
    X₀
end

function _posterior(w::AbstractVector, α::AbstractVector,
                    ϕ::AbstractMatrix, t::AbstractVector)
    w = _newton_method(w,
                       (w) -> _hessian(w, α, ϕ, t),
                       (w) -> _gradient(w, α, ϕ, t),
                       (w) -> _log_posterior(w, α, ϕ, t),
                       )
    @show size(w)
    Σ = inv(-_hessian(w, α, ϕ, t))
    w, Σ
end

SIZE = 2000 #1000 #4300
Train, TrainT = get_data("../data/training.csv", :class)
spec  = RVMSpec(MLKernels.RadialBasisKernel(), 50, identitize, 1e9, 1e-3)
model = fit(spec, Train[1:SIZE, :], TrainT[1:SIZE])

#spec  = RVMSpec(MLKernels.RadialBasisKernel(), 50, whitening, 1e9, 1e-3)
#model = fit(spec, Train[1:SIZE, :], TrainT[1:SIZE])

#Test, TestT = get_data("../data/testing.csv", :class)
#@show predict(model, Test)
#@show TestT
