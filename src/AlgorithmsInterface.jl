module AlgorithmsInterface

abstract type Problem end

abstract type Algorithm end

abstract type State end

function increment!(state::State)
    state.iteration += 1
    return state
end

function solve!(problem::Problem, algorithm::Algorithm, state::State; kwargs...)
    while !is_finished(problem, algorithm, state)
        increment!(state)
        step!(problem, algorithm, state)
    end
    return state
end

function is_finished(problem::Problem, algorithm::Algorithm, state::State)
    return throw(MethodError(is_finished!, (problem, algorithm, state)))
end

function step!(problem::Problem, algorithm::Algorithm, state::State)
    return throw(MethodError(step!, (problem, algorithm, state)))
end

end
