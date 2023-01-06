using CSV
using DataFrames

# from 1 to 13 Rydberg atoms
matrix_sizes = [2^i for i in 1:3:28]

df = DataFrame(
    "Matrix Size" => matrix_sizes,
    "SparseArrays" => missing,
    ["PMCSR $thread Thread" => missing for thread in [2^i for i in 0:7]]...
    )

CSV.write("Benchmark-Data.csv", df)