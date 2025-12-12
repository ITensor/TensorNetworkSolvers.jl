import AlgorithmsInterface as AI
import .AlgorithmsInterfaceExtensions as AIE

#=
    EigenProblem(operator)

Represents the problem we are trying to solve and minimal algorithm-independent
information, so for an eigenproblem it is the operator we want the eigenvector of.
=#
struct EigenProblem{Operator} <: AIE.Problem
    operator::Operator
end

function update!(
        problem::EigenProblem, algorithm::Sweep, state::AI.State
    )
    operator = problem.operator
    x = state.iterate
    region = algorithm.regions[state.iteration]
    region_kwargs = algorithm.region_kwargs(problem, algorithm, state)

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
    x′ = "region = $region" *
        ", update_kwargs = $(region_kwargs.update)" *
        ", insert_kwargs = $(region_kwargs.insert)"
    state.iterate = [x; [x′]]

    return state
end

function dmrg_sweep(operator, state; regions, region_kwargs)
    problem = EigenProblem(operator)
    algorithm = Sweep(; regions, region_kwargs)
    state′ = AI.initialize_state(problem, algorithm; iterate = state)
    AI.solve!(problem, algorithm, state′)
    return state′.iterate
end

function dmrg(operator, state; nsweeps, regions, region_kwargs)
    problem = EigenProblem(operator)
    sweeps = map(1:nsweeps) do i
        return Sweep(; regions, region_kwargs = region_kwargs[i], sweeping_iteration = i)
    end
    algorithm = Sweeping(; sweeps)
    state′ = AI.initialize_state(problem, algorithm; iterate = state)
    AI.solve!(problem, algorithm, state′)
    return state′.iterate
end
