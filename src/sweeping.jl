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

AIE.max_iterations(algorithm::Sweep) = length(algorithm.regions)

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
