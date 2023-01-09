using CSV
using DataFrames

# from 10 to 30 Rydberg atoms, skipping every other number
matrix_sizes = [2^i for i in 10:2:30]

df = DataFrame(
    "Matrix Size" => matrix_sizes,
    "SparseArrays" => missing,
    ["PMCSR $thread Thread" => missing for thread in [2^i for i in 0:7]]...
    )

CSV.write("Benchmark-Data.csv", df)