module NeuralNetworkQMC

using Zygote, Flux
import ChainRulesCore: @ignore_derivatives

abstract type Hamiltonian end


struct HubbardHamiltonian{T} <: Hamiltonian
    t::T
    U::T
end

struct State
    spin_up :: Vector{Int}
    spin_down :: Vector{Int}
end
Base.length(x::State) = length(x.spin_up) 
State(n::Int) = State(zeros(Int, n), zeros(Int, n))
get_n_electrons(x::State) = sum(x.spin_up) + sum(x.spin_down)

include("apply_h.jl")
include("quantum_annealing.jl")
include("qmc_cost.jl")

export State, HubbardHamiltonian, get_n_electrons

end # module NeuralNetworkQMC
