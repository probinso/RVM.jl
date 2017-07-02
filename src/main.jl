using DataFrames
include("RVM.jl")
using .RVM

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


SIZE = 100, 1000 , 2300, 4300
Train, TrainT = get_data("../data/training.csv", :class)

for K in (MLKernels.LinearKernel(), MLKernels.RadialBasisKernel(), MLKernels.GaussianKernel())
    @show "******************"
    @show K
    spec  = RVM.RVMSpec(K, 50, standard_score, 1e5, 1e6)
    for S in SIZE
        @show S
        @time model = RVM.fit(spec, Train[1:S, :], TrainT[1:S])
        @show size(model.RV)
        TrainP = RVM.predict(model, Train)
#=
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
=#
    end
end
