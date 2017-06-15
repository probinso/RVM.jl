using Convex
using MLBase
using MLKernels
using Compat
using DataFrames

type RVMSpec
    kernel::MLKernels.Kernel
    n_iter::Integer
    normalizer::Function
    threshold::Real
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


function get_data(ifname, target)
    data   = DataFrames.readtable(ifname)

    Obs  = convert(Matrix{Float64},
                   data[setdiff(names(data), [target])])
    ObsT = convert(Vector, data[target])
    Obs, ObsT
end

#predict(model, Test)

function fit{R <: Real}(S::RVMSpec, Obs::AbstractMatrix{R}, ObsT::AbstractVector)

    # sample count, features
    n, p = size(Obs)

    normalize = S.normalizer(Obs)
    X = normalize(Obs)

    lm = labelmap(ObsT)
    encode(V) = labelencode(lm, V)
    T  = encode(ObsT)

    ϕ = MLKernels.kernelmatrix(S.kernel, X, X)

    basis_count = size(ϕ, 2)
    α = ones(basis_count) ./ basis_count
    β = 1e-6

    m = zeros(basis_count)

    w = zeros(p)

    modelfit::RVMFit{R} = RVMFit(S.kernel, normalize, encode, false, 0, zeros(p))
end

function _classify(X::AbstractMatrix, ϕ::AbstractMatrix)
    Compat.logistic(ϕ *  X)
end

function _log_posterior(X::AbstractMatrix, α::AbstractVector, ϕ::AbstractMatrix, t::AbstractVector)
    A = diagm(α)
    y = _classify(X, ϕ)
    log_p = 0.5 * X' * A * X - sum(log([ y[t .== 1] (1 - y[t .== 2]) ]))
    jacobian = A * X - ϕ' * (t - y)
    log_p, jacobian
end

function _hessian(X::AbstractMatrix, α::AbstractVector, ϕ::AbstractMatrix, t::AbstractVector)
    A = diagm(α)
    y = _classify(X, ϕ)
    B = diagm(y * (1 - y))

    #∇(w) = ϕ' * (t - y) - A * w
    H = -1 *  (ϕ' * B * ϕ + A)
end

    function _posterior(X::AbstractMatrix, α::AbstractVector, ϕ::AbstractMatrix, t::AbstractVector)
    log_p, jacobian = _log_posterior(X, α, ϕ, t)



#=
def _log_posterior(self, m, alpha, phi, t):

        y = self._classify(m, phi)

        log_p = -1 * (np.sum(np.log(y[t == 1]), 0) +
                      np.sum(np.log(1-y[t == 0]), 0))
        log_p = log_p + 0.5*np.dot(m.T, np.dot(np.diag(alpha), m))

        jacobian = np.dot(np.diag(alpha), m) - np.dot(phi.T, (t-y))

        return log_p, jacobian
=#



Train, TrainT = get_data("../data/training.csv", :class)
spec  = RVMSpec(MLKernels.GaussianKernel(), 10, whitening, 1e-3)
model = fit(spec, Train, TrainT)

#Test, TestT = get_data("../data/testing.csv", :class)



#=
function prune(X, α, ϕ, σ, thresh)
    keep = α .< M.thresh
    X[keep, :], α[keep], ϕ[:, keep], σ...


function fit{R <: Real}
    (M::RVMSpec, Obs::AbstractMatrix{R}, ObsT::AbstractVector)

    # sample count, features
    n, p = size(Obs)

    normalize = M.normalizer(Obs)
    X = normalize(Obs)

    lm = labelmap(ObsT)
    encode(V) = labelencode(lm, V)
    T  = encode(ObsT)

    ϕ = MLKernels.kernelmatrix(M.kernel, X, X)

    basis_count = size(ϕ, 2)
    α = ones(basis_count) ./ basis_count
    β = 1e-6

    m = zeros(basis_count)

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
