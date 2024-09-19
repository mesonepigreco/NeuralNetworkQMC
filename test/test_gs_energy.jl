using Test
using NeuralNetworkQMC
using LinearAlgebra


function test_ground_state_energy()
    # Tets the energy of the state 
    # |up, down> + |full, 0>   
    function ψ(x :: State) :: Complex{Float64}
        ret_val = zero(Complex{Float64})
        if get_n_electrons(x) == 2 

            if x.spin_up[1] == 1 && x.spin_down[1] == 1
                ret_val = one(Complex{Float64})
            end

            if x.spin_up[1] == 1 && x.spin_down[2] == 1
                ret_val = one(Complex{Float64})
            end
        end
        return ret_val
    end


    # set up the initial state
    start_state = State(2)
    start_state.spin_up .= [1, 0]
    start_state.spin_down .= [0, 1]

    ensemble = [State(2) for i in 1:1000]

    # Run a QMC simulation with the given wave function 
    NeuralNetworkQMC.quantum_annealing!(ensemble, start_state, ψ)

    # Check if the distribution is correctly 0.5 0.5 on the two states
    n_full = 0
    n_half = 0
    for i in 1:length(ensemble)
        if ensemble[i].spin_up == [1, 0] && ensemble[i].spin_down == [0, 1]
            n_full += 1
        end

        if ensemble[i].spin_up == [1, 0] && ensemble[i].spin_down == [1, 0]
            n_half += 1
        end
    end

    println("|full, 0> = ", n_full/length(ensemble))
    println("|half, half> = ", n_half/length(ensemble))

    @test n_full + n_half == length(ensemble)
    @test n_full / length(ensemble) ≈ 0.5 atol=0.1
    @test n_half / length(ensemble) ≈ 0.5 atol=0.1

    # Now compute the energy of the state
    t = 0.5
    U = 1.0
    H = HubbardHamiltonian(t, U)
    energy = NeuralNetworkQMC.get_total_energy(ensemble, ψ, H)

    println("Energy = ", energy)
    println("Exact energy = ", 0.5(U - 4t))

    @test energy ≈ 0.5(U - 4t) atol=0.1
end


test_ground_state_energy()
