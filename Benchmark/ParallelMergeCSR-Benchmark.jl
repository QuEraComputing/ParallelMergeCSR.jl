using SparseArrays
using ParallelMergeCSR
using BenchmarkTools
using CSV
using DataFrames
using Bloqade


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

    t = @benchmark ParallelMergeCSR.mul!($C, $A, $B, $α, $β) samples=2000 evals=1

    df = DataFrame(CSV.File("Benchmark-Data.csv"))
    df[!, "PMCSR $(Threads.nthreads()) Thread, $num_atoms atoms"] = t.times
    CSV.write("Benchmark-Data.csv", df)
    
end
