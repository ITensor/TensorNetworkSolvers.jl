module AlgorithmsInterface

abstract type AbstractProblem end

abstract type AbstractAlgorithm end

abstract type AbstractState end

function increment!(state::AbstractState)
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
mutable struct State{Iterate} <: AbstractState
    iterate::Iterate
    iteration::Int
end

function solve!(
        problem::AbstractProblem, algorithm::AbstractAlgorithm, state::AbstractState
    )
    while !is_finished(problem, algorithm, state)
        increment!(state)
        step!(problem, algorithm, state)
    end
    return state
end

function is_finished(
    problem::AbstractProblem, algorithm::AbstractAlgorithm, state::AbstractState)
    return throw(MethodError(is_finished!, (problem, algorithm, state)))
end

function step!(problem::AbstractProblem, algorithm::AbstractAlgorithm, state::AbstractState)
    return throw(MethodError(step!, (problem, algorithm, state)))
end

abstract type AbstractAlgorithmIterator end

function Base.iterate(itr::AbstractAlgorithmIterator, init = nothing)
    is_finished(itr.problem, itr.algorithm, itr.state) && return nothing
    increment!(itr.state)
    step!(itr.problem, itr.algorithm, itr.state)
    return itr.state, nothing
end

struct AlgorithmIterator{Problem, Algorithm, State} <: AbstractAlgorithmIterator
    problem::Problem
    algorithm::Algorithm
    state::State
end

end
