using Plots
using CSV
using DataFrames

# single threaded comparison
# can have single threaded SparseArrays, followed by increasing number of threads afterwards (1, 2, 4, 8)

df = DataFrame(CSV.File("Benchmark-Data.csv"))
x = df[!, "Matrix Size"]
ys = []
labels = String[]
# probably set up a macro to handle this
for col_name in names(df)
    if col_name != "Matrix Size"
        push!(ys, df[!,col_name])
        push!(labels, col_name)
    end
end

plot(x, 
    ys, 
    y_scale=:log10,
    title="Median Times for Matrix-Vector multiplication", 
    label=reshape(labels, 1, length(labels)),
    xlabel="Matrix Size",
    xformatter=:plain,
    ylabel="Time (log(ns))",
    marker=(:circle), 
    legend=:bottomright
)
