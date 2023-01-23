using BenchmarkTools
using ParallelMergeCSR
using SparseArrays
using SparseMatricesCSR
using ThreadedSparseCSR
ThreadedSparseCSR.get_num_threads()

function benchmark(A,x)

    y = similar(x)
    return @benchmark ParallelMergeCSR.mul!($y, $(transpose(A)), $x, 1.0, 0.0)
end

function benchmark_base(A,x)
    y = similar(x)
    return @benchmark SparseArrays.mul!($y, $(transpose(A)), $x, 1.0, 0.0)
end

function benchmark_tmul(csrA,x)
    y = similar(x)
    return @benchmark tmul!($y, $csrA, $x, 1.0, 0.0)
end

function benchmark_bmul(csrA,x)
    y = similar(x)
    return @benchmark bmul!($y, $csrA, $x, 1.0, 0.0)
end

reports = Dict(
    "this" => [],
    "base" => [],
    "tmul" => [],
    "bmul" => [],
)

for n in 4:2:20
    A = sprand(2^n,2^n,n/2^n)
    x = rand(2^n)
    csrA = SparseMatrixCSR(transpose(sprand(2^n, 2^n, 1e-4)));

    @info "benchmarking" n
    push!(reports["this"], benchmark(A,x))
    push!(reports["base"], benchmark_base(A,x))
    push!(reports["tmul"], benchmark_tmul(csrA,x))
    push!(reports["bmul"], benchmark_bmul(csrA,x))
end

speedup = map(reports["base"], reports["this"]) do br, r
    median(br).time / median(r).time
end

speedup = map(reports["base"], reports["tmul"]) do br, r
    median(br).time / median(r).time
end

speedup = map(reports["base"], reports["bmul"]) do br, r
    median(br).time / median(r).time
end