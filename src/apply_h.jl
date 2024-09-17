

@doc raw"""
    local_energy(x :: State, ψ :: Function, H :: Hamiltonian)

Perform the QMC local energy
"""
function local_energy(x :: State, ψ :: Function, H :: HubbardHamiltonian{T}) where T
    new_state = State(length(x))
    
    H_ij = zero(Complex{T})
    this_state = ψ(x)
    for i in 1:length(x)
        # Next site
        i_next = (i - 1) % length(x) + 1
        i_prev = (i + 1) % length(x) + 1

        # Spinup forward hopping
        if x.spin_up[i] == 1 && x.spin_up[i_next] == 0 
            new_state.spin_up[i] = 0 
            new_state.spin_up[i_next] = 1
            H_ij += -H.t * ψ(new_state)
            # Restore state
            new_state.spin_up[i] = 1
            new_state.spin_up[i_next] = 0
        end

        # Spinup backward hopping
        if x.spin_up[i] == 1 && x.spin_up[i_prev] == 0
            new_state.spin_up[i] = 0
            new_state.spin_up[i_prev] = 1
            H_ij += -H.t * ψ(new_state)
            # Restore state
            new_state.spin_up[i] = 1
            new_state.spin_up[i_prev] = 0
        end

        # Spindown forward hopping
        if x.spin_down[i] == 1 && x.spin_down[i_next] == 0
            new_state.spin_down[i] = 0
            new_state.spin_down[i_next] = 1
            H_ij += -H.t * ψ(new_state)
            # Restore state
            new_state.spin_down[i] = 1
            new_state.spin_down[i_next] = 0
        end

        # Spindown backward hopping
        if x.spin_down[i] == 1 && x.spin_down[i_prev] == 0
            new_state.spin_down[i] = 0
            new_state.spin_down[i_prev] = 1
            H_ij += -H.t * ψ(new_state)
            # Restore state
            new_state.spin_down[i] = 1
            new_state.spin_down[i_prev] = 0
        end

        # Add the Hubbard term
        if x.spin_up[i] == 1 && x.spin_down[i] == 1
            H_ij += H.U * this_state
        end
    end

    return H_ij / conj(this_state)
end


@doc raw"""
    get_total_energy(ensemble :: Vector{State}, ψ :: Function, H :: Hamiltonian)
    get_local_energy(ensemble :: Vector{State}, ψ :: Function, H :: Hamiltonian, ψ_gen :: Function)

Calculate the total energy of the ensemble.
If `ψ_gen` is provided, it will be assumed that the ensemble is extracted from the generator `ψ_gen` 
with the importance sampling distribution.

This allows to exploit differentiation.
"""
function get_total_energy(ensemble :: Vector{State}, ψ :: Function, H :: HubbardHamiltonian{T}) where T
    total_energy = zero(Complex{T})
    for x in ensemble
        total_energy += local_energy(x, ψ, H)
    end
    return total_energy / length(ensemble)
end
function get_total_energy(ensemble :: Vector{State}, ψ :: Function, H :: HubbardHamiltonian{T}, ψ_gen :: Function) where T
    total_energy = zero(Complex{T})
    weights = zeros(T, length(ensemble))
    for i in 1:length(ensemble)
        weights[i] = abs2(ψ(ensemble[i])) / abs2(ψ_gen(ensemble[i]))
    end
    # Normalize weights
    weights ./= sum(weights)

    for x in ensemble
        total_energy += local_energy(x, ψ, H) * weights[i] 
    end
    return total_energy
end


