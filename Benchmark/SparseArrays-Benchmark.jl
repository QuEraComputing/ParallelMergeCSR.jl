using SparseArrays
using BenchmarkTools
using CSV
using DataFrames
using Bloqade

# open pre-made CSV
df = DataFrame(CSV.File("Benchmark-Data.csv"))

# based on lattice sizes
sparse_arrays_times = Float64[]

# number of samples (single time/memory observation) to take
BenchmarkTools.DEFAULT_PARAMETERS.samples = 20
# number of evaluation per sample
BenchmarkTools.DEFAULT_PARAMETERS.evals = 100

# extend to 30 atoms for more rigorous testing
for num_atoms in Int.(log2.(df[!,"Matrix Size"]))
    
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
    push!(sparse_arrays_times, median(t).time)
end

# store results
df[!, "SparseArrays"] = sparse_arrays_times

# save file
CSV.write("Benchmark-Data.csv", df)