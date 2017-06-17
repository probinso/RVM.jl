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
        keep[end] = true

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

    modelfit::RVMFit{R} = RVMFit(S.kernel, normalize, encode, succ, steps, w)
end

function _classify(w::AbstractVector, ϕ::AbstractMatrix)
    [expit(x) for x in ϕ * w]
end

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

#=
def _log_posterior(self, m, alpha, phi, t):

        y = self._classify(w, phi)

        log_p = -1 * (np.sum(np.log(y[t == 1]), 0) +
                      np.sum(np.log(1-y[t == 0]), 0))
        log_p = log_p + 0.5*np.dot(w.T, np.dot(np.diag(alpha), m))

        jacobian = np.dot(np.diag(alpha), m) - np.dot(phi.T, (t-y))

        return log_p, jacobian
=#

Train, TrainT = get_data("../data/training.csv", :class)
spec  = RVMSpec(MLKernels.GaussianKernel(), 10, whitening, 1e9, 1e-3)
model = fit(spec, Train, TrainT)

#Test, TestT = get_data("../data/testing.csv", :class)

#=
function prune(X, α, ϕ, σ, thresh)
    keep = α .< M.thresh
    X[keep, :], α[keep], ϕ[:, keep], σ...


function fit{R <: Real}
    (w::RVMSpec, Obs::AbstractMatrix{R}, ObsT::AbstractVector)

    # sample count, features
    n, p = size(Obs)

    normalize = M.normalizer(Obs)
    X = normalize(Obs)

    lm = labelmap(ObsT)
    encode(V) = labelencode(lm, V)
    T  = encode(ObsT)

    ϕ = MLKernels.kernelmatrix(w.kernel, X, X)

    basis_count = size(ϕ, 2)
    α = ones(basis_count) ./ basis_count
    β = 1e-6

    w = zeros(basis_count)

    for i in 1:M.n_iter

    end
end


function unknown(X, α, ϕ, t)
    A = diagm(α)
    y = classify(X, ϕ)
    B = diagm(y * (1 - y))
    C = B + ϕ * A * ϕ'

    ∇(w) = ϕ' * (t - y) - A * w
    ∇∇   = -1 *  (ϕ' * B * ϕ + A)

    w★ = inv(A) * ϕ' * (t - y)
    Σ  = inv(ϕ' * B * ϕ + A)

    [(1 - α[i] * diag(Σ)[i]) / (w★[i] ^ 2) for i in 1:size(α, 1)]
end


IRLS(w, H,

function gradient(X, α, ϕ, t)
    A = diagm(α)

∇  = gradient(X, α, ϕ, t)

    = hessian(X, α, ϕ, t)

posterior(

=#
