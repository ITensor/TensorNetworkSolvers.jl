module AlgorithmsInterfaceExtensions

import AlgorithmsInterface as AI

#========================== Patches for AlgorithmsInterface.jl ============================#

abstract type Problem <: AI.Problem end
abstract type Algorithm <: AI.Algorithm end
abstract type State <: AI.State end

function AI.initialize_state!(
        problem::Problem, algorithm::Algorithm, state::State; iterate = nothing
    )
    !isnothing(iterate) && (state.iterate = iterate)
    AI.initialize_state!(
        problem, algorithm, algorithm.stopping_criterion, state.stopping_criterion_state
    )
    return state
end

function AI.initialize_state(
        problem::Problem, algorithm::Algorithm; kwargs...
    )
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion
    )
    return DefaultState(; stopping_criterion_state, kwargs...)
end

#============================ DefaultState ================================================#

@kwdef mutable struct DefaultState{
        Iterate, StoppingCriterionState <: AI.StoppingCriterionState,
    } <: State
    iterate::Iterate
    iteration::Int = 0
    stopping_criterion_state::StoppingCriterionState
end

#============================ increment! ==================================================#

# Custom version of `increment!` that also takes the problem and algorithm as arguments.
function AI.increment!(problem::Problem, algorithm::Algorithm, state::State)
    return AI.increment!(state)
end

#============================ solve! ======================================================#

# Custom version of `solve!` that allows specifying the logger and also overloads
# `increment!` on the problem and algorithm.
function basetypenameof(x)
    return Symbol(last(split(String(Symbol(Base.typename(typeof(x)).wrapper)), ".")))
end
function default_logging_context_prefix(problem::Problem, algorithm::Algorithm)
    return Symbol(basetypenameof(problem), :_, basetypenameof(algorithm), :_)
end
function AI.solve!(
        problem::Problem, algorithm::Algorithm, state::State;
        logging_context_prefix = default_logging_context_prefix(problem, algorithm),
        kwargs...,
    )
    logger = AI.algorithm_logger()

    context_suffixes = [:Start, :PreStep, :PostStep, :Stop]
    contexts = Dict(context_suffixes .=> Symbol.(logging_context_prefix, context_suffixes))

    # initialize the state and emit message
    AI.initialize_state!(problem, algorithm, state; kwargs...)
    AI.emit_message(logger, problem, algorithm, state, contexts[:Start])

    # main body of the algorithm
    while !AI.is_finished!(problem, algorithm, state)
        AI.increment!(problem, algorithm, state)

        # logging event between convergence check and algorithm step
        AI.emit_message(logger, problem, algorithm, state, contexts[:PreStep])

        # algorithm step
        AI.step!(problem, algorithm, state; logging_context_prefix)

        # logging event between algorithm step and convergence check
        AI.emit_message(logger, problem, algorithm, state, contexts[:PostStep])
    end

    # emit message about finished state
    AI.emit_message(logger, problem, algorithm, state, contexts[:Stop])
    return state
end

function AI.solve(
        problem::Problem, algorithm::Algorithm;
        logging_context_prefix = default_logging_context_prefix(problem, algorithm),
        kwargs...,
    )
    state = AI.initialize_state(problem, algorithm; kwargs...)
    return AI.solve!(problem, algorithm, state; logging_context_prefix, kwargs...)
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
    return AI.increment!(iterator.problem, iterator.algorithm, iterator.state)
end
function AI.step!(iterator::AlgorithmIterator)
    return AI.step!(iterator.problem, iterator.algorithm, iterator.state)
end
function Base.iterate(iterator::AlgorithmIterator, init = nothing)
    AI.is_finished!(iterator) && return nothing
    AI.increment!(iterator)
    AI.step!(iterator)
    return iterator.state, nothing
end

#============================ with_algorithmlogger ========================================#

# Allow passing functions, not just CallbackActions.
@inline function with_algorithmlogger(f, args::Pair{Symbol, AI.LoggingAction}...)
    return AI.with_algorithmlogger(f, args...)
end
@inline function with_algorithmlogger(f, args::Pair{Symbol}...)
    return AI.with_algorithmlogger(f, (first.(args) .=> AI.CallbackAction.(last.(args)))...)
end

#============================ NestedAlgorithm =============================================#

#=
    NestedAlgorithm(sweeps::AbstractVector{<:Algorithm})

An algorithm that consists of running an algorithm at each iteration
from a list of stored algorithms.
=#
@kwdef struct NestedAlgorithm{
        Algorithms <: AbstractVector{<:Algorithm},
        StoppingCriterion <: AI.StoppingCriterion,
    } <: Algorithm
    algorithms::Algorithms
    stopping_criterion::StoppingCriterion = AI.StopAfterIteration(length(algorithms))
end
function NestedAlgorithm(f::Function, nalgorithms::Int; kwargs...)
    return NestedAlgorithm(; algorithms = f.(1:nalgorithms), kwargs...)
end

max_iterations(algorithm::NestedAlgorithm) = length(algorithm.algorithms)

function AI.step!(
        problem::AI.Problem, algorithm::NestedAlgorithm, state::AI.State;
        logging_context_prefix = Symbol()
    )
    # Perform the current sweep.
    sub_algorithm = algorithm.algorithms[state.iteration]
    sub_state = AI.initialize_state(problem, sub_algorithm; state.iterate)
    logging_context_prefix = Symbol(logging_context_prefix, :Sweep_)
    AI.solve!(problem, sub_algorithm, sub_state; logging_context_prefix)
    state.iterate = sub_state.iterate
    return state
end

#============================ FlattenedAlgorithm ==========================================#

# Flatten a nested algorithm.
@kwdef struct FlattenedAlgorithm{
        Algorithms <: AbstractVector{<:Algorithm},
        StoppingCriterion <: AI.StoppingCriterion,
    } <: Algorithm
    algorithms::Algorithms
    stopping_criterion::StoppingCriterion =
        AI.StopAfterIteration(sum(max_iterations, algorithms))
end
function FlattenedAlgorithm(f::Function, nalgorithms::Int; kwargs...)
    return FlattenedAlgorithm(; algorithms = f.(1:nalgorithms), kwargs...)
end

@kwdef mutable struct FlattenedAlgorithmState{
        Iterate, StoppingCriterionState <: AI.StoppingCriterionState,
    } <: State
    iterate::Iterate
    iteration::Int = 0
    parent_iteration::Int = 1
    child_iteration::Int = 0
    stopping_criterion_state::StoppingCriterionState
end

function AI.initialize_state(
        problem::Problem, algorithm::FlattenedAlgorithm; kwargs...
    )
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion
    )
    return FlattenedAlgorithmState(; stopping_criterion_state, kwargs...)
end
function AI.increment!(
        problem::Problem, algorithm::Algorithm, state::FlattenedAlgorithmState
    )
    # Increment the total iteration count.
    state.iteration += 1
    if state.child_iteration ≥ max_iterations(algorithm.algorithms[state.parent_iteration])
        # We're on the last iteration of the child algorithm, so move to the next
        # child algorithm.
        state.parent_iteration += 1
        state.child_iteration = 1
    else
        # Iterate the child algorithm.
        state.child_iteration += 1
    end
    return state
end
function AI.step!(
        problem::AI.Problem, algorithm::FlattenedAlgorithm, state::FlattenedAlgorithmState;
        logging_context_prefix = Symbol()
    )
    algorithm_sweep = algorithm.algorithms[state.parent_iteration]
    state_sweep = AI.initialize_state(
        problem, algorithm_sweep;
        state.iterate, iteration = state.child_iteration
    )
    AI.step!(problem, algorithm_sweep, state_sweep; logging_context_prefix)
    state.iterate = state_sweep.iterate
    return state
end

end
