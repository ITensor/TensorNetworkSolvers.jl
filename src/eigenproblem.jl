import AlgorithmsInterface as AI
import .AlgorithmsInterfaceExtensions as AIE

function dmrg_sweep(operator, state; regions, region_kwargs)
    problem = EigenProblem(operator)
    algorithm = Sweep(; regions, region_kwargs)
    return AI.solve(problem, algorithm; iterate = state).iterate
end

function dmrg(operator, state; nsweeps, regions, region_kwargs, kwargs...)
    problem = EigenProblem(operator)
    algorithm = Sweeping(nsweeps) do i
        return Sweep(; regions, region_kwargs = region_kwargs[i])
    end
    return AI.solve(problem, algorithm; iterate = state, kwargs...).iterate
end

#=
    EigenProblem(operator)

Represents the problem we are trying to solve and minimal algorithm-independent
information, so for an eigenproblem it is the operator we want the eigenvector of.
=#
struct EigenProblem{Operator} <: AIE.Problem
    operator::Operator
end

function AI.step!(problem::EigenProblem, algorithm::Sweep, state::AI.State; kwargs...)
    iterate = solve_region!!(problem, algorithm.algorithms[state.iteration], state.iterate)
    state.iterate = iterate
    return state
end

# extract!, update!, insert! for the region.
function solve_region!!(problem::EigenProblem, algorithm::RegionAlgorithm, state)
    operator = problem.operator
    region = algorithm.region
    region_kwargs = algorithm.kwargs(algorithm, state)

    #=
    # Reduce the `operator` and state `x` onto the region `region`,
    # and call `eigsolve` on the reduced operator and state using the
    # keyword arguments determined from `region_kwargs`.
    operator_region = reduced_operator(operator, x, region)
    x_region = reduced_state(x, region)
    x_region′ = eigsolve(operator_region, x_region; region_kwargs.update...)
    x′ = insert(x, region, x_region′; region_kwargs.insert...)
    state.state = x′
    =#

    # Dummy update for demonstration purposes.
    state′ = "region = $region" *
        ", update_kwargs = $(region_kwargs.update)" *
        ", insert_kwargs = $(region_kwargs.insert)"
    state = [state; [state′]]

    return state
end
