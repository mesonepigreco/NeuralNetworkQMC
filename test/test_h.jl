using NeuralNetworkQMC
using Test


function test_hamiltonian()

    function ψ_hopping(x :: State) :: Complex{Float64}
        if sum(x.spin_down) > 1
            return 0.0
        end
        if x.spin_up[1]* x.spin_up[2] == 0 && sum(x.spin_up) == 1
            return 1.0
        end
        0.0
    end

    # H local should be t
    hamiltonian = HubbardHamiltonian(1.0, 1.0)

    my_state = State(2)
    my_state.spin_up[1] = 1
    other_state = State(2)
    other_state.spin_up[2] = 1
    println("ψ(x) = $(ψ_hopping(my_state))")
    println("ψ2(x) = $(ψ_hopping(other_state))")

    energy = NeuralNetworkQMC.local_energy(my_state, ψ_hopping, hamiltonian)

    println("Energy: $energy")
    println("Hopping: $(hamiltonian.t)")
end


test_hamiltonian()
