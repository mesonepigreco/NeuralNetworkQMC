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

include("apply_h.jl")
include("quantum_annealing.jl")

export State, HubbardHamiltonian

end # module NeuralNetworkQMC
