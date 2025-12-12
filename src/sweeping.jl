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
    sweeping_iteration::Int = 0
    stopping_criterion::StoppingCriterion = AI.StopAfterIteration(length(regions))
end
function Sweep(
        regions::Vector,
        region_kwargs::NamedTuple,
        sweeping_iteration::Int,
        stopping_criterion::AI.StoppingCriterion,
    )
    return Sweep(regions, Returns(region_kwargs), sweeping_iteration, stopping_criterion)
end

maxiter(algorithm::Sweep) = length(algorithm.regions)

function AI.step!(
        problem::AI.Problem, algorithm::Sweep, state::AI.State
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

# TODO: Use a proper stopping criterion.
function AI.is_finished(
        problem::AI.Problem, algorithm::Sweep, state::AI.State
    )
    state.iteration == 0 && return false
    return state.iteration >= length(algorithm.regions)
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
    sweeps = f.(1:nsweeps)
    return Sweeping(; sweeps, kwargs...)
end

maxiter(algorithm::Sweeping) = length(algorithm.sweeps)

function AI.step!(
        problem::AI.Problem, algorithm::Sweeping, state::AI.State
    )
    # Perform the current sweep.
    algorithm_sweep = algorithm.sweeps[state.iteration]
    state_sweep = AI.initialize_state(problem, algorithm_sweep; iterate = state.iterate)
    AI.solve!(problem, algorithm_sweep, state_sweep)
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
struct ByRegion{Algorithm <: Sweeping} <: AIE.Algorithm
    parent::Algorithm
end
## function AI.initialize_state(
##         problem::AI.Problem, algorithm::ByRegion, x
##     )
##     return AI.State(x, (; sweep = 1, region = 0))
## end
function AI.is_finished(
        problem::AI.Problem, algorithm::ByRegion, state::AI.State
    )
    sweep_iteration = state.iteration.sweep
    region_iteration = state.iteration.region
    return sweep_iteration ≥ maxiter(algorithm.parent) &&
        region_iteration ≥ maxiter(algorithm.parent.sweeps[sweep_iteration])
end
function AI.increment!(
        problem::AI.Problem, algorithm::ByRegion, state::AI.State
    )
    sweep_iteration = state.iteration.sweep
    region_iteration = state.iteration.region
    if region_iteration < maxiter(algorithm.parent.sweeps[sweep_iteration])
        region_iteration += 1
    else
        sweep_iteration += 1
        region_iteration = 1
    end
    state.iteration = (; sweep = sweep_iteration, region = region_iteration)
    return state
end
function AI.step!(problem::AI.Problem, algorithm::ByRegion, state::AI.State)
    sweep = algorithm.parent.sweeps[state.iteration.sweep]
    sweep_state = AI.State(state.iterate, state.iteration.region)
    AI.step!(problem, sweep, sweep_state)
    state.iterate = sweep_state.iterate
    return state
end
