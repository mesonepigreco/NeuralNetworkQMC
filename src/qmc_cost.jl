function run_qmc(n_sites :: Int, H :: Hamiltonian;
        n_samples :: Int = 100, 
        η = 0.1,
        n_optimization_steps = 1000,
        qmc_keys...)
    # Create the initial wavefunction

    ψ = Dense(2n_sites, 2, tanh)


    function loss(ψ)
        state = State(n_sites)

        ensemble = [State(n_sites) for i in 1:n_samples]
        @ignore_derivatives quantum_annealing!(ensemble, ψ, H; qmc_keys...)

        # Use the importance sampling for the gradient
        get_total_energy(ensemble, ψ, H, ψ)
    end

    opt = ADAM(η)
    for i in 1:n_optimization_steps
        ∇ = gradient(()->loss(ψ), Flux.params(ψ))
        Flux.update!(opt, ψ, ∇)
    end

    return ψ
end
