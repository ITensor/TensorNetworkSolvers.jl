using BenchmarkTools
using TensorNetworkSolvers

SUITE = BenchmarkGroup()
SUITE["rand"] = @benchmarkable rand(10)

# Write your benchmarks here.
