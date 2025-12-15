import AlgorithmsInterface as AI
using Graphs: path_graph
using TensorNetworkSolvers: ByRegion, EigenProblem, Sweeping, Sweep, dmrg, dmrg_sweep
import TensorNetworkSolvers.AlgorithmsInterfaceExtensions as AIE
using Test: @test, @testset

@testset "TensorNetworkSolvers" begin
    @testset "dmrg_sweep" begin
        operator = path_graph(4)
        regions = [(1, 2), (2, 3), (3, 4)]
        tol = 1.0e-4
        maxdim = 50
        region_kwargs = (;
            update = (; tol),
            insert = (; maxdim),
        )
        state = []
        x = dmrg_sweep(operator, state; regions, region_kwargs)
        @test length(x) == 3
    end
    @testset "dmrg" begin
        operator = path_graph(4)
        regions = [(1, 2), (2, 3), (3, 4)]
        nsweeps = 3
        tols = [1.0e-3, 1.0e-4, 1.0e-5]
        maxdims = [20, 50, 100]
        region_kwargs = map(1:nsweeps) do i
            return (;
                update = (; tol = tols[i]),
                insert = (; maxdim = maxdims[i]),
            )
        end
        state = []
        x = dmrg(operator, state; nsweeps, regions, region_kwargs)
        @test length(x) == nsweeps * length(regions)
    end
    @testset "dmrg: region-dependent kwargs" begin
        operator = path_graph(4)
        regions = [(1, 2), (2, 3), (3, 4)]
        nsweeps = 3
        tols = [1.0e-3, 1.0e-4, 1.0e-5]
        maxdims = [20, 50, 100]
        region_kwargs = map(1:nsweeps) do i
            return function (problem, alg, region_state)
                return (;
                    update = (; tol = tols[i] / region_state.iteration),
                    insert = (; maxdim = maxdims[i] * region_state.iteration),
                )
            end
        end
        state = []
        x = dmrg(operator, state; nsweeps, regions, region_kwargs)
        @test length(x) == nsweeps * length(regions)
    end
    @testset "iterate" begin
        operator = path_graph(4)
        regions = [(1, 2), (2, 3), (3, 4)]
        nsweeps = 3
        tols = [1.0e-3, 1.0e-4, 1.0e-5]
        maxdims = [20, 50, 100]
        region_kwargs = map(1:nsweeps) do i
            return (;
                update = (; tol = tols[i]),
                insert = (; maxdim = maxdims[i]),
            )
        end
        x = []
        problem = EigenProblem(operator)
        algorithm = Sweeping(nsweeps) do i
            Sweep(; regions, region_kwargs = region_kwargs[i])
        end
        state = AI.initialize_state(problem, algorithm; iterate = x)
        iterator = AIE.algorithm_iterator(problem, algorithm, state)
        iterations = Int[]
        for state in iterator
            push!(iterations, state.iteration)
        end
        @test iterations == 1:nsweeps
        @test length(state.iterate) == nsweeps * length(regions)
    end
    @testset "ByRegion" begin
        operator = path_graph(4)
        regions = [(1, 2), (2, 3), (3, 4)]
        nsweeps = 3
        tols = [1.0e-3, 1.0e-4, 1.0e-5]
        maxdims = [20, 50, 100]
        region_kwargs = map(1:nsweeps) do i
            return (;
                update = (; tol = tols[i]),
                insert = (; maxdim = maxdims[i]),
            )
        end
        x = []
        problem = EigenProblem(operator)
        sweeping = Sweeping(nsweeps) do i
            Sweep(; regions, region_kwargs = region_kwargs[i])
        end
        algorithm = ByRegion(; sweeping)
        state = AI.initialize_state(problem, algorithm; iterate = x)
        iterator = AIE.algorithm_iterator(problem, algorithm, state)
        iterations = Int[]
        for state in iterator
            push!(iterations, state.iteration)
        end
        @test iterations == 1:(nsweeps * length(regions))
        @test length(state.iterate) == nsweeps * length(regions)
    end
end
