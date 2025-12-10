using Graphs: path_graph
using TensorNetworkSolvers: dmrg, dmrg_sweep
using Test: @test, @testset

@testset "TensorNetworkSolvers" begin
    # Perform a single sweep.
    operator = path_graph(4)
    regions = [(1, 2), (2, 3), (3, 4)]
    tol = 1.0e-4
    maxdim = 50
    region_kwargs = (;
        update = (; tol),
        insert = (; maxdim),
    )
    x1 = dmrg_sweep(operator; regions, region_kwargs)
    @test length(x1) == 3

    # Sweep-dependent region kwargs (uniform across regions).
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
    x2 = dmrg(operator; nsweeps, regions, region_kwargs)
    @test length(x2) == 3 * 3

    # Region-dependent kwargs.
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
    x3 = dmrg(operator; nsweeps, regions, region_kwargs)
    @test length(x3) == 3 * 3

end
