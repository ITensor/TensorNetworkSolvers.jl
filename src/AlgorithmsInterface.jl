module AlgorithmsInterface

abstract type Problem end

abstract type Algorithm end

abstract type State end

function increment!(
        problem::Problem, algorithm::Algorithm, state::State
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
mutable struct DefaultState{Iterate, Iteration} <: State
    iterate::Iterate
    iteration::Iteration
end

function initialize_state(
        problem::Problem, algorithm::Algorithm, x
    )
    return DefaultState(x, 0)
end

using Base.ScopedValues: ScopedValue, with
const CALLBACKS = ScopedValue(Dict{Symbol, Any}())
function callback(
        problem::Problem,
        algorithm::Algorithm,
        state::State,
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
        problem::Problem, algorithm::Algorithm, state::State
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
        problem::Problem, algorithm::Algorithm, state::State
    )
    return is_finished(
        problem, algorithm, state, state.stopping_criterion, state.stopping_criterion_state
    )
end

function step!(problem::Problem, algorithm::Algorithm, state::State)
    return throw(MethodError(step!, (problem, algorithm, state)))
end

abstract type AlgorithmIterator end

struct DefaultAlgorithmIterator{Problem, Algorithm, State} <: AlgorithmIterator
    problem::Problem
    algorithm::Algorithm
    state::State
end
function iterator(
        problem::Problem, algorithm::Algorithm, state::State
    )
    return DefaultAlgorithmIterator(problem, algorithm, state)
end

function is_finished(itr::AlgorithmIterator)
    return is_finished(itr.problem, itr.algorithm, itr.state)
end
function callback(itr::AlgorithmIterator, event::Symbol)
    return callback(itr.problem, itr.algorithm, itr.state, event)
end
function increment!(itr::AlgorithmIterator)
    return increment!(itr.problem, itr.algorithm, itr.state)
end
function step!(itr::AlgorithmIterator)
    return step!(itr.problem, itr.algorithm, itr.state)
end

function Base.iterate(itr::AlgorithmIterator, init = nothing)
    is_finished(itr) && return nothing
    callback(itr, :PreStep)
    increment!(itr)
    step!(itr)
    callback(itr, :PostStep)
    return itr.state, nothing
end

abstract type StoppingCriterion end
abstract type StoppingCriterionState end

end
