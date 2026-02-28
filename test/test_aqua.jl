using Aqua: Aqua
using TensorNetworkSolvers: TensorNetworkSolvers
using Test: @testset

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(TensorNetworkSolvers)
end
