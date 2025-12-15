module AlgorithmsInterfaceExtensions

import AlgorithmsInterface as AI

#========================== Patches for AlgorithmsInterface.jl ============================#

abstract type Problem <: AI.Problem end
abstract type Algorithm <: AI.Algorithm end
abstract type State <: AI.State end

#============================ DefaultState ================================================#

@kwdef mutable struct DefaultState{
        Iterate, StoppingCriterionState <: AI.StoppingCriterionState,
    } <: State
    iterate::Iterate
    iteration::Int = 0
    stopping_criterion_state::StoppingCriterionState
end
function AI.initialize_state(
        problem::AI.Problem, algorithm::Algorithm; kwargs...
    )
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion
    )
    return DefaultState(; stopping_criterion_state, kwargs...)
end
function AI.initialize_state!(
        problem::AI.Problem, algorithm::AI.Algorithm, state::DefaultState; iterate = nothing
    )
    !isnothing(iterate) && (state.iterate = iterate)
    AI.initialize_state!(
        problem, algorithm, algorithm.stopping_criterion, state.stopping_criterion_state
    )
    return state
end

#============================ increment! ==================================================#

# Custom version of `increment!` that also takes the problem and algorithm as arguments.
function increment!(problem::AI.Problem, algorithm::AI.Algorithm, state::State)
    return AI.increment!(state)
end

#============================ solve! ======================================================#

# Custom version of `solve!` that allows specifying the logger and also overloads
# `increment!` on the problem and algorithm.
function solve!(
        problem::Problem, algorithm::Algorithm, state::State;
        logger = algorithm_logger(), kwargs...,
    )
    # initialize the state and emit message
    AI.initialize_state!(problem, algorithm, state; kwargs...)
    AI.emit_message(logger, problem, algorithm, state, :Start)

    # main body of the algorithm
    while !AI.is_finished!(problem, algorithm, state)
        # logging event between convergence check and algorithm step
        AI.emit_message(logger, problem, algorithm, state, :PreStep)

        # algorithm step
        increment!(problem, algorithm, state)
        AI.step!(problem, algorithm, state)

        # logging event between algorithm step and convergence check
        AI.emit_message(logger, problem, algorithm, state, :PostStep)
    end

    # emit message about finished state
    AI.emit_message(logger, problem, algorithm, state, :Stop)

    return state
end

#============================ AlgorithmIterator ===========================================#

abstract type AlgorithmIterator end

struct DefaultAlgorithmIterator{Problem, Algorithm, State} <: AlgorithmIterator
    problem::Problem
    algorithm::Algorithm
    state::State
end
function algorithm_iterator(
        problem::Problem, algorithm::Algorithm, state::State
    )
    return DefaultAlgorithmIterator(problem, algorithm, state)
end

function AI.is_finished!(iterator::AlgorithmIterator)
    return AI.is_finished!(iterator.problem, iterator.algorithm, iterator.state)
end
function AI.is_finished(iterator::AlgorithmIterator)
    return AI.is_finished(iterator.problem, iterator.algorithm, iterator.state)
end
function increment!(iterator::AlgorithmIterator)
    return increment!(iterator.problem, iterator.algorithm, iterator.state)
end
function AI.step!(iterator::AlgorithmIterator)
    return AI.step!(iterator.problem, iterator.algorithm, iterator.state)
end
function Base.iterate(iterator::AlgorithmIterator, init = nothing)
    AI.is_finished!(iterator) && return nothing
    increment!(iterator)
    AI.step!(iterator)
    return iterator.state, nothing
end

end
