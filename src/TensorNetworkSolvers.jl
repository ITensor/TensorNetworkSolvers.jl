module TensorNetworkSolvers

export dmrg, dmrg_sweep

import AlgorithmsInterface as AI

struct EigenProblem{Operator} <: AI.Problem
    operator::Operator
end

struct DMRGSweep{Regions <: AbstractVector, RegionKwargs <: Function} <: AI.Algorithm
    regions::Regions
    region_kwargs::RegionKwargs
end
function DMRGSweep(regions::AbstractVector, region_kwargs::NamedTuple)
    function region_kwargs_fn(problem::AI.Problem, algorithm::AI.Algorithm, state::AI.State)
        return region_kwargs
    end
    return DMRGSweep(regions, region_kwargs_fn)
end

mutable struct RegionState{TensorNetwork} <: AI.State
    state::TensorNetwork
    iteration::Int
end

function AI.initialize_state(
        problem::EigenProblem, algorithm::DMRGSweep; kwargs...
    )

    # Dummy empty initialization for demonstration purposes.
    # In practice we might randomly initialize a tensor network
    # using information from `problem.operator`.
    x0 = []

    return RegionState(x0, 0)
end

function AI.initialize_state!(
        problem::EigenProblem, algorithm::DMRGSweep, state::RegionState; kwargs...
    )
    # reset the state for the algorithm
    state.state = []
    state.iteration = 0
    return state
end

function AI.step!(
        problem::EigenProblem, algorithm::DMRGSweep, state::RegionState
    )
    operator = problem.operator
    x = state.state
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
    =#

    # Dummy update for demonstration purposes.
    x′ = "region = $region" *
        ", update_kwargs = $(region_kwargs.update)" *
        ", insert_kwargs = $(region_kwargs.insert)"
    state.state = [x; [x′]]

    return state
end

function AI.increment!(state::RegionState)
    state.iteration += 1
    return state
end

function AI.is_finished!(
        problem::EigenProblem, algorithm::DMRGSweep, state::RegionState
    )
    state.iteration == 0 && return false
    return state.iteration >= length(algorithm.regions)
end

function dmrg_sweep(operator; regions, region_kwargs)
    prob = EigenProblem(operator)
    alg = DMRGSweep(regions, region_kwargs)
    state = solve(prob, alg)
    return state.state
end

struct DMRG{Sweeps <: AbstractVector{<:DMRGSweep}} <: AI.Algorithm
    sweeps::Sweeps
end

mutable struct SweepState{TensorNetwork} <: AI.State
    state::TensorNetwork
    iteration::Int
end

function AI.initialize_state(
        problem::EigenProblem, algorithm::DMRG; kwargs...
    )

    # Dummy empty initialization for demonstration purposes.
    # In practice we might randomly initialize a tensor network
    # using information from `problem.operator`.
    x0 = []

    return SweepState(x0, 0)
end

function AI.initialize_state!(
        problem::EigenProblem, algorithm::DMRG, state::SweepState; kwargs...
    )
    # reset the state for the algorithm
    state.state = []
    state.iteration = 0
    return state
end

function AI.step!(
        problem::EigenProblem, algorithm::DMRG, state::SweepState
    )
    # Perform the current DMRG sweep.
    sweep = algorithm.sweeps[state.iteration]
    x = state.state
    region_state = RegionState(x, 0)
    while !AI.is_finished!(problem, sweep, region_state)
        AI.increment!(region_state)
        AI.step!(problem, sweep, region_state)
    end
    state.state = region_state.state
    return state
end

function AI.increment!(state::SweepState)
    state.iteration += 1
    return state
end

function AI.is_finished!(
        problem::EigenProblem, algorithm::DMRG, state::SweepState
    )
    state.iteration == 0 && return false
    return state.iteration >= length(algorithm.sweeps)
end

function dmrg(operator; nsweeps, regions, region_kwargs)
    prob = EigenProblem(operator)
    sweeps = [DMRGSweep(regions, kws) for kws in region_kwargs]
    alg = DMRG(sweeps)
    state = AI.solve(prob, alg)
    return state.state
end

end
