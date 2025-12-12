module AlgorithmsInterfaceExtensions

import AlgorithmsInterface as AI

#========================== Patches for AlgorithmsInterface.jl ============================#

abstract type Problem <: AI.Problem end
abstract type Algorithm <: AI.Algorithm end
abstract type State <: AI.State end

function AI.initialize_state!(
        problem::Problem, algorithm::Algorithm, state::State
    )
    return state
end

#============================ DefaultState ================================================#

@kwdef mutable struct DefaultState{
        Iterate, StoppingCriterionState <: AI.StoppingCriterionState,
    } <: State
    iterate::Iterate
    iteration::Int = 0
    stopping_criterion_state::StoppingCriterionState
end
function AI.initialize_state(
        problem::Problem, algorithm::Algorithm; kwargs...
    )
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion
    )
    return DefaultState(; stopping_criterion_state, kwargs...)
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
function AI.increment!(iterator::AlgorithmIterator)
    return AI.increment!(iterator.state)
end
function AI.step!(iterator::AlgorithmIterator)
    return AI.step!(iterator.problem, iterator.algorithm, iterator.state)
end
function AI.emit_message(iterator::AlgorithmIterator, event::Symbol)
    return AI.emit_message(iterator.problem, iterator.algorithm, iterator.state, event)
end
function AI.emit_message(
        logger::AI.AlgorithmLogger, iterator::AlgorithmIterator, event::Symbol
    )
    return AI.emit_message(
        logger, iterator.problem, iterator.algorithm, iterator.state, event
    )
end
function AI.emit_message(
        logger::Nothing, iterator::AlgorithmIterator, event::Symbol
    )
    return AI.emit_message(
        logger, iterator.problem, iterator.algorithm, iterator.state, event
    )
end

function Base.iterate(iterator::AlgorithmIterator, init = nothing)
    logger = AI.algorithm_logger()
    AI.is_finished!(iterator) && return nothing
    AI.emit_message(logger, iterator, :PreStep)
    AI.increment!(iterator)
    AI.step!(iterator)
    AI.emit_message(logger, iterator, :PostStep)
    return iterator.state, nothing
end

end
