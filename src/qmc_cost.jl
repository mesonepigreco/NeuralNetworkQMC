@doc """
    get_qmc_cost_function(n_sites :: Int, n_electrons :: Int, H :: Hamiltonian, n_samples :: Int; qmc_keys...) :: Function

Create a cost function for the QMC calculation. 
The returned loss function takes a State as input and returns the total energy of the given Hamiltonian for that state.
"""
function get_qmc_cost_function(n_sites :: Int, n_electrons :: Int, H :: Hamiltonian,
        n_samples :: Int; qmc_keys...) :: Function

    function loss(ψ)
        # Initialize the state
        state = State(n_sites)
        # Initialize with the correct number of electrons
        state.spin_down[1:n_electrons÷2] .= 1
        state.spin_up[1:n_electrons - n_electrons÷2] .= 1

        ensemble = [State(n_sites) for i in 1:n_samples]
    
        @info "Annealing"
        quantum_annealing!(ensemble, state, ψ; qmc_keys...)
        # Print the value of the probability
        println("ψ(state): ")
        all_mod = abs2.(ψ.(ensemble))

        for i in 1:length(ensemble)
            println("$i    up: $(ensemble[i].spin_up) down: $(ensemble[i].spin_down)   psi: $(ψ(ensemble[i]))  prob:$(all_mod[i]/sum(all_mod))")
        end

        # Use the importance sampling for the gradient
        get_total_energy(ensemble, ψ, H, ψ)
    end

    return loss
end

export get_qmc_cost_function
