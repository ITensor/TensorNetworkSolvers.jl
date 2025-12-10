import .AlgorithmsInterface as AI

#=
    StateAndIteration(state, iteration::Int)

The "state", which stores both the tensor network state (the `iterate`) and the current
`iteration`, which is the integer corresponding to which region or sweep we are on
(`which_region` or `which_sweep` in ITensorNetworks.jl). For `alg::Sweep`, the
current region is `alg.regions[iteration]`, while for `alg::Sweeping`, the current sweep is
`alg.sweeps[iteration]`.
=#
mutable struct StateAndIteration{Iterate} <: AI.State
    iterate::Iterate
    iteration::Int
end

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
struct Sweep{Regions <: AbstractVector, RegionKwargs <: Function} <: AI.Algorithm
    regions::Regions
    region_kwargs::RegionKwargs
end
function Sweep(regions::AbstractVector, region_kwargs::NamedTuple)
    function region_kwargs_fn(problem::AI.Problem, algorithm::AI.Algorithm, state::AI.State)
        return region_kwargs
    end
    return Sweep(regions, region_kwargs_fn)
end

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
    Sweeping(sweeps::AbstractVector{<:Sweep})

The sweeping algorithm, which just stores a list of sweeps defined above. 
=#
struct Sweeping{Sweeps <: AbstractVector{<:Sweep}} <: AI.Algorithm
    sweeps::Sweeps
end

function AI.step!(
        problem::AI.Problem, algorithm::Sweeping, state::AI.State
    )
    # Perform the current sweep.
    sweep = algorithm.sweeps[state.iteration]
    x = state.iterate
    region_state = StateAndIteration(x, 0)
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
