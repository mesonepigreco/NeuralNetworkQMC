@doc raw"""
    quantum_annealing!(ensemble :: Vector{State}, state :: State, ψ :: Function;
    thermalization_steps :: Int = 1000, n_steps_between_samples :: Int = 100)


Perform a quantum annealing algorithm and fill the ensemble with states
distributed with the probability given by the wave function ψ.
"""
function quantum_annealing!(ensemble :: Vector{State}, state :: State, ψ :: Function; 
                            thermalization_steps :: Int = 1000, n_steps_between_samples :: Int = 100)

    tmp_state = State(length(state))
    tmp_state.spin_up .= state.spin_up
    tmp_state.spin_down .= state.spin_down

    # Allowed moves are: flip a spin, swap two states
    for i in 1:thermalization_steps
        for j in 1:length(state)
            qmc_move!(state, tmp_state, ψ)
        end
    end

    # Store the first configuration
    ensemble[1].spin_up .= state.spin_up
    ensemble[1].spin_down .= state.spin_down

    for i in 2:length(ensemble)
        for j in 1:n_steps_between_samples
            for k in 1:length(state)
                qmc_move!(state, tmp_state, ψ)
            end
        end
        ensemble[i].spin_up .= state.spin_up
        ensemble[i].spin_down .= state.spin_down
    end
end


@doc raw"""
    qmc_move!(x :: State, xtmp :: State, ψ :: Function)

Perform a quantum Monte Carlo move on the state x.
The ``xtmp`` state is used as a temporary state.
"""
function qmc_move!(state :: State, tmp_state :: State, ψ :: Function)
    site = rand(1:length(state))
    site2 = rand(1:(length(state)-1))
    if site2 >= site
        site2 += 1
    end

    # Swap two states
    spin_move = rand(1:2)
    if spin_move == 1
        tmp_state.spin_up[site] = state.spin_up[site2]
        tmp_state.spin_up[site2] = state.spin_up[site]
    else
        tmp_state.spin_down[site] = state.spin_down[site2]
        tmp_state.spin_down[site2] = state.spin_down[site]
    end

    # Calculate the acceptance probability
    p = abs(ψ(tmp_state) / ψ(state))^2
    if rand() < p
        # println("Accepted: $(tmp_state.spin_up) $(tmp_state.spin_down)")
        state.spin_up .= tmp_state.spin_up
        state.spin_down .= tmp_state.spin_down
    else
        tmp_state.spin_up .= state.spin_up
        tmp_state.spin_down .= state.spin_down
    end
end
