using Convex
using MLBase
using MLKernels
using Compat
using DataFrames

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

function get_data(ifname::AbstractString, target::Symbol)
    data   = DataFrames.readtable(ifname)

    Obs  = convert(Matrix{Float64},
                   data[setdiff(names(data), [target])])
    ObsT = convert(Vector, data[target])
    Obs, ObsT
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
    encode(V) = labelencode(lm, V)
    t  = encode(ObsT)

    # design matrix / basis functions
    _ = MLKernels.kernelmatrix(S.kernel, X, X)
    ϕ = [_ ones(size(_, 1))] # augment with bias
    basis_count = size(ϕ, 2)

    # initialize priors
    α₀ = ones(basis_count) .* 1e-6
    α  = α₀[1:end]
    w  = zeros(basis_count)

    # reporting information
    steps = S.n_iter
    succ  = false

    for i in 1:S.n_iter
        println(size(α))

        # update values
        w, Σ = _posterior(w, α, ϕ, t)
        γ = 1 .- α .* diag(Σ)
        α = γ ./ (w .^ 2)

        # identify informative vectors
        keep = α .< S.threshold
        if !any(keep)
            keep[1] = true
        end
        keep[end] = true # save bias

        # downselect uninformative vectors
        α₀ = α₀[keep]
        α  = α[keep]
        γ  = γ[keep]
        ϕ  = ϕ[:, keep]
        Σ  = Σ[keep, keep]
        w  = w[keep]
        RV = RV[keep[1:end-1], :]

        #
        if maximum(abs(α .- α₀)) < S.tol
            steps = i
            succ  = true
            break
        end
        α₀ = α[1:end]
    end

    modelfit::RVMFit{R} = RVMFit(S.kernel, normalize, encode, succ, steps, w, RV)
end

@inline _classify(w::AbstractVector, ϕ::AbstractMatrix) =
    [expit(x) for x in ϕ * w]

function _log_posterior(w::AbstractMatrix, α::AbstractVector,
                        ϕ::AbstractMatrix, t::AbstractVector)
    A = diagm(α)
    y = _classify(w, ϕ)
    log_p = 0.5 * m' * A * w - sum(log([ y[t .== 1] (1 - y[t .== 2]) ]))
    jacobian = A * w - ϕ' * (t - y)
    log_p, jacobian
end

function _hessian(w::AbstractVector, α::AbstractVector,
                  ϕ::AbstractMatrix, t::AbstractVector)
    A = diagm(α)
    y = _classify(w, ϕ)
    B = diagm(y .* (1 - y))

    #∇(w) = ϕ' * (t - y) - A * w
    H = -1 *  (ϕ' * B * ϕ + A)
end

function _posterior(w::AbstractVector, α::AbstractVector,
                    ϕ::AbstractMatrix, t::AbstractVector)
    _log_posterior
    _hessian
    #(w, α, ϕ, t)
    w = w #... # minimize weights with fixed priors
    Σ = inv(_hessian(w, α, ϕ, t))
    w, Σ
end

Train, TrainT = get_data("../data/training.csv", :class)
spec  = RVMSpec(MLKernels.GaussianKernel(), 10, whitening, 1e9, 1e-3)
model = fit(spec, Train, TrainT)

#Test, TestT = get_data("../data/testing.csv", :class)
