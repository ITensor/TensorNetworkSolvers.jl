module TensorNetworkSolvers

export dmrg, dmrg_sweep

import AlgorithmsInterface as AI

#=
    EigenProblem(operator)

Represents the problem we are trying to solve and minimal algorithm-independent
information, so for an eigenproblem it is the operator we want the eigenvector of.
=#
struct EigenProblem{Operator} <: AI.Problem
    operator::Operator
end

#=
    StateAndIteration(state, iteration::Int)

The "state", which stores both the tensor network state and the current iteration,
which is the integer corresponding to which region or sweep we are on (`which_region`
or `which_sweep` in ITensorNetworks.jl). For `alg::DMRGSweep`, the current region is
`alg.regions[iteration]`, while for `alg::DMRG`, the current sweep is
`alg.sweeps[iteration]`.
=#
mutable struct StateAndIteration{State} <: AI.State
    state::State
    iteration::Int
end

function AI.increment!(state::StateAndIteration)
    state.iteration += 1
    return state
end


#=
    DMRGSweep(regions::AbsractVector, region_kwargs::Function)
    DMRGSweep(regions::AbsractVector, region_kwargs::NamedTuple)

The "algorithm" for performing a single DMRG sweep over a list of regions. It also
store a function that takes the problem, algorithm, and state (tensor network, current
region, etc.) and returns keyword arguments for performing the region update on the
current region. For simplicity, it also accepts a `NamedTuple` of keyword arguments
which is converted into a function that always returns the same keyword arguments
for an region.
=#
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

function AI.initialize_state(
        problem::EigenProblem, algorithm::DMRGSweep; kwargs...
    )

    # Dummy empty initialization for demonstration purposes.
    # In practice we might randomly initialize a tensor network
    # using information from `problem.operator`.
    x0 = []

    return StateAndIteration(x0, 0)
end

function AI.initialize_state!(
        problem::EigenProblem, algorithm::DMRGSweep, state::StateAndIteration; kwargs...
    )
    # reset the state for the algorithm
    state.state = []
    state.iteration = 0
    return state
end

function AI.step!(
        problem::EigenProblem, algorithm::DMRGSweep, state::StateAndIteration
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
    state.state = x′
    =#

    # Dummy update for demonstration purposes.
    x′ = "region = $region" *
        ", update_kwargs = $(region_kwargs.update)" *
        ", insert_kwargs = $(region_kwargs.insert)"
    state.state = [x; [x′]]

    return state
end

function AI.is_finished!(
        problem::EigenProblem, algorithm::DMRGSweep, state::StateAndIteration
    )
    state.iteration == 0 && return false
    return state.iteration >= length(algorithm.regions)
end

function dmrg_sweep(operator; regions, region_kwargs)
    prob = EigenProblem(operator)
    alg = DMRGSweep(regions, region_kwargs)
    state = AI.solve(prob, alg)
    return state.state
end

#=
    DMRG(sweeps::AbstractVector{<:DMRGSweep})

The DMRG algorithm, which just stores a list of DMRG sweeps defined above. 
=#
struct DMRG{Sweeps <: AbstractVector{<:DMRGSweep}} <: AI.Algorithm
    sweeps::Sweeps
end

function AI.initialize_state(
        problem::EigenProblem, algorithm::DMRG; kwargs...
    )

    # Dummy empty initialization for demonstration purposes.
    # In practice we might randomly initialize a tensor network
    # using information from `problem.operator`.
    x0 = []

    return StateAndIteration(x0, 0)
end

function AI.initialize_state!(
        problem::EigenProblem, algorithm::DMRG, state::StateAndIteration; kwargs...
    )
    # reset the state for the algorithm
    state.state = []
    state.iteration = 0
    return state
end

function AI.step!(
        problem::EigenProblem, algorithm::DMRG, state::StateAndIteration
    )
    # Perform the current DMRG sweep.
    sweep = algorithm.sweeps[state.iteration]
    x = state.state
    region_state = StateAndIteration(x, 0)
    while !AI.is_finished!(problem, sweep, region_state)
        AI.increment!(region_state)
        AI.step!(problem, sweep, region_state)
    end
    state.state = region_state.state
    return state
end

function AI.is_finished!(
        problem::EigenProblem, algorithm::DMRG, state::StateAndIteration
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
