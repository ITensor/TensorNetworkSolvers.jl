import AlgorithmsInterface as AI
import .AlgorithmsInterfaceExtensions as AIE

#=
    Sweep(regions::AbsractVector, region_kwargs::Function, iteration::Int = 0)
    Sweep(regions::AbsractVector, region_kwargs::NamedTuple, iteration::Int = 0)

The "algorithm" for performing a single sweep over a list of regions. It also
stores a function that takes the problem, algorithm, and state (tensor network, current
region, etc.) and returns keyword arguments for performing the region update on the
current region. For simplicity, it also accepts a `NamedTuple` of keyword arguments
which is converted into a function that always returns the same keyword arguments
for an region.
=#
@kwdef struct Sweep{
        Regions <: Vector, RegionKwargs <: Function, StoppingCriterion <: AI.StoppingCriterion,
    } <: AIE.Algorithm
    regions::Regions
    region_kwargs::RegionKwargs
    stopping_criterion::StoppingCriterion = AI.StopAfterIteration(length(regions))
end
function Sweep(
        regions::Vector,
        region_kwargs::NamedTuple,
        stopping_criterion::AI.StoppingCriterion,
    )
    return Sweep(regions, Returns(region_kwargs), stopping_criterion)
end

maxiter(algorithm::Sweep) = length(algorithm.regions)

function AI.step!(
        problem::AI.Problem, algorithm::Sweep, state::AI.State; kwargs...
    )
    extract!(problem, algorithm, state)
    update!(problem, algorithm, state)
    insert!(problem, algorithm, state)
    return state
end

function extract!(
        problem::AI.Problem, algorithm::Sweep, state::AI.State
    )
    # Extraction step goes here.
    return state
end
function update!(
        problem::AI.Problem, algorithm::Sweep, state::AI.State
    )
    # Update step goes here.
    return state
end
function insert!(
        problem::AI.Problem, algorithm::Sweep, state::AI.State
    )
    # Insert step goes here.
    return state
end

#=
    Sweeping(sweeps::Vector{<:Sweep})

The sweeping algorithm, which just stores a list of sweeps defined above. 
=#
@kwdef struct Sweeping{
        Sweeps <: Vector{<:Sweep}, StoppingCriterion <: AI.StoppingCriterion,
    } <: AIE.Algorithm
    sweeps::Sweeps
    stopping_criterion::StoppingCriterion = AI.StopAfterIteration(length(sweeps))
end
function Sweeping(f::Function, nsweeps::Int; kwargs...)
    return Sweeping(; sweeps = f.(1:nsweeps), kwargs...)
end

maxiter(algorithm::Sweeping) = length(algorithm.sweeps)
nregions(algorithm::Sweeping) = sum(maxiter, algorithm.sweeps)

function AI.step!(
        problem::AI.Problem, algorithm::Sweeping, state::AI.State;
        logging_context_prefix = Symbol()
    )
    # Perform the current sweep.
    algorithm_sweep = algorithm.sweeps[state.iteration]
    state_sweep = AI.initialize_state(problem, algorithm_sweep; state.iterate)
    logging_context_prefix = Symbol(logging_context_prefix, :Sweep_)
    AI.solve!(problem, algorithm_sweep, state_sweep; logging_context_prefix)
    state.iterate = state_sweep.iterate
    return state
end

# TODO: Use a proper stopping criterion.
function AI.is_finished(
        problem::AI.Problem, algorithm::Sweeping, state::AI.State
    )
    state.iteration == 0 && return false
    return state.iteration >= length(algorithm.sweeps)
end

# Sweeping by region.
@kwdef struct ByRegion{
        ParentAlgorithm <: Sweeping, StoppingCriterion <: AI.StoppingCriterion,
    } <: AIE.Algorithm
    sweeping::ParentAlgorithm
    stopping_criterion::StoppingCriterion = AI.StopAfterIteration(nregions(sweeping))
end

@kwdef mutable struct ByRegionState{
        Iterate, StoppingCriterionState <: AI.StoppingCriterionState,
    } <: AIE.State
    iterate::Iterate
    iteration::Int = 0
    sweeping_iteration::Int = 1
    sweep_iteration::Int = 0
    stopping_criterion_state::StoppingCriterionState
end

function AI.initialize_state(
        problem::AIE.Problem, algorithm::ByRegion; kwargs...
    )
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion
    )
    return ByRegionState(; stopping_criterion_state, kwargs...)
end
function AI.increment!(problem::AIE.Problem, algorithm::AIE.Algorithm, state::ByRegionState)
    # Increment the total iteration count.
    state.iteration += 1
    if state.sweep_iteration ≥ maxiter(algorithm.sweeping.sweeps[state.sweeping_iteration])
        # We're on the last region of the sweep, so move to the next sweep.
        state.sweeping_iteration += 1
        state.sweep_iteration = 1
    else
        # Move to the next region in the current sweep.
        state.sweep_iteration += 1
    end
    return state
end
function AI.step!(
        problem::AI.Problem, algorithm::ByRegion, state::ByRegionState;
        logging_context_prefix = Symbol()
    )
    algorithm_sweep = algorithm.sweeping.sweeps[state.sweeping_iteration]
    state_sweep = AI.initialize_state(
        problem, algorithm_sweep;
        state.iterate, iteration = state.sweep_iteration
    )
    AI.step!(problem, algorithm_sweep, state_sweep; logging_context_prefix)
    state.iterate = state_sweep.iterate
    return state
end
