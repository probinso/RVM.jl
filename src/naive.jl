using Convex
using MLBase
using MLKernels

type RVMSpec
    kernel::MLKernels.Kernel
    degree::Integer
    n_iter::Integer
    normalizer::Function
    threshold::Real
end

type RVMFit{R <: Real}
    kernel::MLKernels.Kernel
    normalize::Function
    encode::Function
    converged::Bool
    w::Vector{R}
end

function whitening(Obs::AbstractMatrix)
    μ = mean(Obs, 1)
    σ = var(Obs, 1)

    f(X) = (X .- μ) ./ σ
end

function fit{R <: Real}
    (M::RVMSpec, Obs::AbstractMatrix{R}, ObsT::AbstractVector)

    # sample count, features
    n, p = size(Obs)

    normalize = M.normalizer(Obs)
    X = normalize(Obs)

    lm = labelmap(ObsT)
    encode(V) = labelencode(lm, V)
    T  = encode(ObsT)
    #T[T .== 2] .= -1
    #T[T .== 1] .=  1

    ϕ = MLKernels.kernelmatrix(M.kernel, X, X)
    basis_count = size(ϕ, 1)
    α = ones(basis_count) ./ basis_count
    β = 1e-6

    for i in 1:M.n_iter
        
    end
end
