import .AlgorithmsInterface as AI

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
@kwdef struct Sweep{Regions <: Vector, RegionKwargs <: Function} <: AI.Algorithm
    regions::Regions
    region_kwargs::RegionKwargs
    iteration::Int = 0
end
function Sweep(regions::Vector, region_kwargs::NamedTuple, iteration::Int = 0)
    function region_kwargs_fn(
            problem::AI.Problem,
            algorithm::AI.Algorithm,
            state::AI.State,
        )
        return region_kwargs
    end
    return Sweep(regions, region_kwargs_fn, iteration)
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
struct Sweeping{Sweeps <: Vector{<:Sweep}} <: AI.Algorithm
    sweeps::Sweeps
end

maxiter(algorithm::Sweeping) = length(algorithm.sweeps)

function AI.step!(
        problem::AI.Problem, algorithm::Sweeping, state::AI.State
    )
    # Perform the current sweep.
    sweep = algorithm.sweeps[state.iteration]
    x = state.iterate
    region_state = AI.initialize_state(problem, sweep, x)
    AI.solve!(problem, sweep, region_state)
    state.iterate = region_state.iterate
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
struct ByRegion{Algorithm <: Sweeping} <: AI.Algorithm
    parent::Algorithm
end
function AI.initialize_state(
        problem::AI.Problem, algorithm::ByRegion, x
    )
    return AI.State(x, (; sweep = 1, region = 0))
end
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
