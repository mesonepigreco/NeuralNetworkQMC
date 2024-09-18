using Flux
using NeuralNetworkQMC
using LinearAlgebra
using ReverseDiff
using DiffResults

function solve_2site()
    N_sites = 2
    N_electrons = 2
    N_samples = 20

    N_thermalization = 100
    N_steps = 100

    # Define the Hamiltonian (t = 1.0, U = 0.0)
    H = HubbardHamiltonian(1.0, 0.0)

    # Get the cost function
    loss = get_qmc_cost_function(N_sites, N_electrons, H, N_samples;
                                thermalization_steps = N_thermalization, 
                                n_steps_between_samples = N_steps)

    # Create the model wavefunction
    ψ_model = Dense(2*N_sites => 2, tanh)
    
    # Destructure the model
    θ, func = Flux.destructure(ψ_model)

    # Define the loss as a function of the parameters
    function my_loss(params)
        println("params: ", params)
        @show params
        println("type(params): ", typeof(params))
        function ψ(x :: State) :: Complex
            x_ = [x.spin_up; x.spin_down]
            my_ψ = func(params)
            my_ψ_val = my_ψ(x_)
            return my_ψ_val[1] + im * my_ψ_val[2]
        end

        loss(ψ)
    end

    # Prepare the gradient tape
    results = similar(θ)
    all_results = DiffResults.GradientResult(results)
    f_tape = ReverseDiff.GradientTape(my_loss, θ)
    compiled_f_tape = ReverseDiff.compile(f_tape)

    opt = ADAM(0.1)
    for i in 1:100
        # Calculate the gradient
        
        #total_energy, back = Zygote.pullback(my_loss, θ)
        #grad = back(1)
        ReverseDiff.gradient!(all_results, compiled_f_tape, θ)

        total_enerty = DiffResults.value(all_results)
        grad = DiffResults.gradient(all_results)

        println("$i  $total_energy  $(norm(grad))")

        # Update the parameters
        Flux.update!(opt, θ, grad)
    end
end

solve_2site()

