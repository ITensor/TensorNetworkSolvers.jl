import AlgorithmsInterface as AI
import .AlgorithmsInterfaceExtensions as AIE

@kwdef struct RegionAlgorithm{Region, RegionKwargs <: Function} <: AIE.Algorithm
    region::Region
    region_kwargs::RegionKwargs
end
function RegionAlgorithm(region, region_kwargs::NamedTuple)
    return RegionAlgorithm(region, Returns(region_kwargs))
end

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
struct Sweep{
        Algorithms <: AbstractVector, StoppingCriterion <: AI.StoppingCriterion,
    } <: AIE.Algorithm
    algorithms::Algorithms
    stopping_criterion::StoppingCriterion
end
function Sweep(;
        regions::AbstractVector, region_kwargs,
        stopping_criterion::AI.StoppingCriterion = AI.StopAfterIteration(length(regions)),
    )
    algorithms = map(regions) do region
        return RegionAlgorithm(region, region_kwargs)
    end
    return Sweep(algorithms, stopping_criterion)
end
AIE.max_iterations(algorithm::Sweep) = length(algorithm.algorithms)
