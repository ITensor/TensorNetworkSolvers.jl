module TensorNetworkSolvers

export dmrg, dmrg_sweep

import AlgorithmsInterface as AI

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

function AI.initialize_state!(
        problem::AI.Problem, algorithm::Sweep, state::AI.State; kwargs...
    )
    # Defined as a no-op so it isn't called in `AI.solve!`.
    return state
end

# TODO: Use a proper stopping criterion.
function AI.is_finished!(
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

function AI.initialize_state!(
        problem::AI.Problem, algorithm::Sweeping, state::AI.State; kwargs...
    )
    # Defined as a no-op so it isn't called in `AI.solve!`.
    return state
end

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
function AI.is_finished!(
        problem::AI.Problem, algorithm::Sweeping, state::AI.State
    )
    state.iteration == 0 && return false
    return state.iteration >= length(algorithm.sweeps)
end

#=
    EigenProblem(operator)

Represents the problem we are trying to solve and minimal algorithm-independent
information, so for an eigenproblem it is the operator we want the eigenvector of.
=#
struct EigenProblem{Operator} <: AI.Problem
    operator::Operator
end

function AI.step!(
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
    prob = EigenProblem(operator)
    alg = Sweep(regions, region_kwargs)
    state′ = StateAndIteration(state, 0)
    AI.solve!(prob, alg, state′)
    return state′.iterate
end

function dmrg(operator, state; nsweeps, regions, region_kwargs)
    prob = EigenProblem(operator)
    sweeps = [Sweep(regions, kws) for kws in region_kwargs]
    alg = Sweeping(sweeps)
    state′ = StateAndIteration(state, 0)
    AI.solve!(prob, alg, state′)
    return state′.iterate
end

end
