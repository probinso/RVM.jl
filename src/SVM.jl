module SVM

using StatsBase
using MLKernels
using MLBase


type SVMFit{T <: Real}
    Normalizer::Function
    Converged::Bool
    pass::Integer
    w::Vector{T}
end


function svm_qp{T <: Real}
    (X::AbstractMatrix{T},
     Y::AbstractVector{T},
     k::Integer     = 5,
     λ::Real        = 0.1,
     count::Integer = 100,
     )
    # p features, n observations
    p, n = size(X)

    # initialize weights
    w::Vector{T} = randn(p)
    
    for t in 1:count
        
