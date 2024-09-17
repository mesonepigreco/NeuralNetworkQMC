using NeuralNetworkQMC
using Test


function test_hamiltonian()

    function ψ_hopping(x :: State)
        if sum(x.spin_down) > 1
            return 0
        end
        if x.spin_up[1]* x.spin_up[2] == 1
            return 1
        end
    end

    # H local should be t
    hamiltonian = HubbardHamiltonian(1.0, 1.0)

    my_state = State(2)
    my_state.spin_up[1] = 1

    energy = NeuralNetworkQMC.local_energy(my_state, ψ, hamiltonian)

    println("Energy: $energy")
    println("Hopping: $(hamiltonian.t)")
end


test_hamiltonian()
