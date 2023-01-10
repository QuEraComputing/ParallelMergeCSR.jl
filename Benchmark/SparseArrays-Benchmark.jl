using SparseArrays
using BenchmarkTools
using CSV
using DataFrames
using Bloqade

# open pre-made CSV
df = DataFrame(CSV.File("Benchmark-Data.csv"))

# number of samples (single time/memory observation) to take
BenchmarkTools.DEFAULT_PARAMETERS.samples = 2000
# number of evaluation per sample
BenchmarkTools.DEFAULT_PARAMETERS.evals = 1

# first columns are all sparseArrays
for num_atoms in 10:2:30
    
    println("Number of Atoms: $num_atoms")
    # create lattice
    atoms = generate_sites(ChainLattice(), num_atoms, scale=6.3)
    # create Hamiltonian
    h = rydberg_h(atoms; Δ=1.2*2π, Ω=1.1*2π)
    # convert to sparse matrix with adjoint
    A = transpose(mat(Float64, h))
    # random vector to multiply with A
    B = rand(size(A, 2))
    # store result
    C = zeros(ComplexF64, size(B))
    α = 1.0
    β = 1.0

    t = @benchmark SparseArrays.mul!($C, $A, $B, $α, $β)

    df[!, "SparseArrays $num_atoms atoms"] = t.times
    CSV.write("Benchmark-Data.csv", df)
end
