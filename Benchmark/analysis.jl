using Plots
using CSV
using DataFrames
using Statistics

# open CSV as Dataframe
df = DataFrame(CSV.File("Benchmark-Data.csv"))

# clean up, drop any column with "30 atoms", tests kept getting killed by Linux
# (probably due to memory)
select!(df, Not(r"30"))

# extract minimum and median values (originally 2000 runs for each column)
min_df = mapcols(col -> minimum(col), df)
# median_df = mapcols(col -> median(col), df)

# split between SparseArrays and Parallel Merge
sparse_arrays_df = select(min_df, r"SparseArrays")
parallel_merge_df = select(min_df, r"PMCSR")

# get sparse_arrays_df data from dataframe for Plots.jl
sparse_arrays_times = collect(values(sparse_arrays_df[1,:]))

# empty vector, store multiple NTuples of data for each thread from
parallel_merge_times = []
parallel_merge_labels = String[]
# split parallel merge into 1,2...,128 threads respectively
for i in 0:7
    label = "PMCSR $(2^i) Thread"
    push!(parallel_merge_labels, label)
    push!(parallel_merge_times, select(parallel_merge_df, Regex(label))[1,:] |> values |> collect)
end

ys = [sparse_arrays_times, parallel_merge_times...]
labels = ["SparseArrays",parallel_merge_labels...]
# probably set up a macro to handle this

# x axis
x = collect(10:2:28)

plot(x, 
     ys, 
     y_scale=:log10,
     title="Minimum Times for Matrix-Vector multiplication", 
     label=reshape(labels, 1, length(labels)),
     xlabel="# Atoms in Chain",
     xformatter=:plain,
     ylabel="Time (log(ns))",
     marker=(:x), 
     legend=:bottomright
)
