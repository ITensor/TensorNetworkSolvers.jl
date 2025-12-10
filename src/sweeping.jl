import .AlgorithmsInterface as AI

#=
    Sweep(regions::AbsractVector, region_kwargs::Function)
    Sweep(regions::AbsractVector, region_kwargs::NamedTuple)

The "algorithm" for performing a single sweep over a list of regions. It also
stores a function that takes the problem, algorithm, and state (tensor network, current
region, etc.) and returns keyword arguments for performing the region update on the
current region. For simplicity, it also accepts a `NamedTuple` of keyword arguments
which is converted into a function that always returns the same keyword arguments
for an region.
=#
struct Sweep{Regions <: AbstractVector, RegionKwargs <: Function} <: AI.AbstractAlgorithm
    regions::Regions
    region_kwargs::RegionKwargs
end
function Sweep(regions::AbstractVector, region_kwargs::NamedTuple)
    function region_kwargs_fn(
            problem::AI.AbstractProblem,
            algorithm::AI.AbstractAlgorithm,
            state::AI.AbstractState,
        )
        return region_kwargs
    end
    return Sweep(regions, region_kwargs_fn)
end

maxiter(algorithm::Sweep) = length(algorithm.regions)

function AI.step!(
        problem::AI.AbstractProblem, algorithm::Sweep, state::AI.AbstractState
    )
    extract!(problem, algorithm, state)
    update!(problem, algorithm, state)
    insert!(problem, algorithm, state)
    return state
end

function extract!(
        problem::AI.AbstractProblem, algorithm::Sweep, state::AI.AbstractState
    )
    # Extraction step goes here.
    return state
end
function update!(
        problem::AI.AbstractProblem, algorithm::Sweep, state::AI.AbstractState
    )
    # Update step goes here.
    return state
end
function insert!(
        problem::AI.AbstractProblem, algorithm::Sweep, state::AI.AbstractState
    )
    # Insert step goes here.
    return state
end

# TODO: Use a proper stopping criterion.
function AI.is_finished(
        problem::AI.AbstractProblem, algorithm::Sweep, state::AI.AbstractState
    )
    state.iteration == 0 && return false
    return state.iteration >= length(algorithm.regions)
end

#=
    Sweeping(sweeps::AbstractVector{<:Sweep})

The sweeping algorithm, which just stores a list of sweeps defined above. 
=#
struct Sweeping{Sweeps <: AbstractVector{<:Sweep}} <: AI.AbstractAlgorithm
    sweeps::Sweeps
end

maxiter(algorithm::Sweeping) = length(algorithm.sweeps)

function AI.step!(
        problem::AI.AbstractProblem, algorithm::Sweeping, state::AI.AbstractState
    )
    # Perform the current sweep.
    sweep = algorithm.sweeps[state.iteration]
    x = state.iterate
    region_state = AI.State(x, 0)
    AI.solve!(problem, sweep, region_state)
    state.iterate = region_state.iterate
    return state
end

# TODO: Use a proper stopping criterion.
function AI.is_finished(
        problem::AI.AbstractProblem, algorithm::Sweeping, state::AI.AbstractState
    )
    state.iteration == 0 && return false
    return state.iteration >= length(algorithm.sweeps)
end

# Sweeping by region.
struct ByRegion{Algorithm <: Sweeping} <: AI.AbstractAlgorithm
    algorithm::Algorithm
end
function AI.is_finished(
        problem::AI.AbstractProblem, algorithm::ByRegion, state::AI.AbstractState
    )
    sweep_iteration = state.iteration.sweep
    region_iteration = state.iteration.region
    return sweep_iteration ≥ maxiter(algorithm.algorithm) &&
        region_iteration ≥ maxiter(algorithm.algorithm.sweeps[sweep_iteration])
end
function AI.increment!(
        problem::AI.AbstractProblem, algorithm::ByRegion, state::AI.AbstractState
    )
    sweep_iteration = state.iteration.sweep
    region_iteration = state.iteration.region
    if region_iteration < maxiter(algorithm.algorithm.sweeps[sweep_iteration])
        region_iteration += 1
    else
        sweep_iteration += 1
        region_iteration = 1
    end
    state.iteration = (; sweep = sweep_iteration, region = region_iteration)
    return state
end
function AI.step!(problem::AI.AbstractProblem, algorithm::ByRegion, state::AI.AbstractState)
    sweep = algorithm.algorithm.sweeps[state.iteration.sweep]
    sweep_state = AI.State(state.iterate, state.iteration.region)
    AI.step!(problem, sweep, sweep_state)
    state.iterate = sweep_state.iterate
    return state
end
