module AlgorithmsInterface

abstract type AbstractProblem end

abstract type AbstractAlgorithm end

abstract type AbstractState end

function increment!(
        problem::AbstractProblem, algorithm::AbstractAlgorithm, state::AbstractState
    )
    state.iteration += 1
    return state
end

#=
    State(state, iteration::Int)

The "state", which stores both the tensor network state (the `iterate`) and the current
`iteration`, which is the integer corresponding to which region or sweep we are on
(`which_region` or `which_sweep` in ITensorNetworks.jl). For `alg::Sweep`, the
current region is `alg.regions[iteration]`, while for `alg::Sweeping`, the current sweep is
`alg.sweeps[iteration]`.
=#
mutable struct State{Iterate, Iteration} <: AbstractState
    iterate::Iterate
    iteration::Iteration
end

function initialize_state(
        problem::AbstractProblem, algorithm::AbstractAlgorithm, x
    )
    return State(x, 0)
end

using Base.ScopedValues: ScopedValue, with
const CALLBACKS = ScopedValue(Dict{Symbol, Any}())
function callback(
        problem::AbstractProblem,
        algorithm::AbstractAlgorithm,
        state::AbstractState,
        event::Symbol,
    )
    f = get(CALLBACKS[], event, Returns(nothing))
    f(problem, algorithm, state)
    return nothing
end
function with_callbacks(f, callbacks::Pair{Symbol}...)
    return with(f, CALLBACKS => Dict(callbacks...))
end

function solve!(
        problem::AbstractProblem, algorithm::AbstractAlgorithm, state::AbstractState
    )
    callback(problem, algorithm, state, :Start)
    while !is_finished(problem, algorithm, state)
        callback(problem, algorithm, state, :PreStep)
        increment!(problem, algorithm, state)
        step!(problem, algorithm, state)
        callback(problem, algorithm, state, :PostStep)
    end
    callback(problem, algorithm, state, :Stop)
    return state
end

function is_finished(
        problem::AbstractProblem, algorithm::AbstractAlgorithm, state::AbstractState
    )
    return throw(MethodError(is_finished, (problem, algorithm, state)))
end

function step!(problem::AbstractProblem, algorithm::AbstractAlgorithm, state::AbstractState)
    return throw(MethodError(step!, (problem, algorithm, state)))
end

abstract type AbstractAlgorithmIterator end

struct AlgorithmIterator{Problem, Algorithm, State} <: AbstractAlgorithmIterator
    problem::Problem
    algorithm::Algorithm
    state::State
end
function iterator(
        problem::AbstractProblem, algorithm::AbstractAlgorithm, state::AbstractState
    )
    return return AlgorithmIterator(problem, algorithm, state)
end

function is_finished(itr::AbstractAlgorithmIterator)
    return is_finished(itr.problem, itr.algorithm, itr.state)
end
function callback(itr::AbstractAlgorithmIterator, event::Symbol)
    return callback(itr.problem, itr.algorithm, itr.state, event)
end
function increment!(itr::AbstractAlgorithmIterator)
    return increment!(itr.problem, itr.algorithm, itr.state)
end
function step!(itr::AbstractAlgorithmIterator)
    return step!(itr.problem, itr.algorithm, itr.state)
end

function Base.iterate(itr::AbstractAlgorithmIterator, init = nothing)
    is_finished(itr) && return nothing
    callback(itr, :PreStep)
    increment!(itr)
    step!(itr)
    callback(itr, :PostStep)
    return itr.state, nothing
end

end
