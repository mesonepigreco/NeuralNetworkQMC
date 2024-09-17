module NeuralNetworkQMC

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
State(n::Int) = State(zero(Int, n), zero(Int, n))

include("apply_h.jl")
include("quantum_annealing.jl")


end # module NeuralNetworkQMC
