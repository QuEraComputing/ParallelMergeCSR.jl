using CSV
using DataFrames

num_atoms = 10:2:30
num_evals = 2000

# should have 2000 rows
# SparseArrays 11 Atoms, 12 Atoms...
# PMCSR # threads 11 Atoms, 12 Atoms 
df = DataFrame(
    ["SparseArrays $atoms atoms"  => zeros(Float64, num_evals) for atoms in num_atoms]...,
    ["PMCSR $(2^i) Thread, $atoms atoms" => zeros(Float64, num_evals) for i in 0:7 for atoms in num_atoms]...,
    #["PMCSR $(2^i) Thread, $i atoms" => [[Vector{Float64}() for i in collect(length(num_atoms))] for i in 0:7] for i in num_atoms]...
)

CSV.write("Benchmark-Data.csv", df)