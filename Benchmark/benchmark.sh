
echo "Creating Benchmarks CSV"
julia --project=. prepare_csv.jl

#echo "Benchmarking SparseArrays"
#julia --project=. SparseArrays-Benchmark.jl

for i in {0..7}
do
    echo "Benchmarking against $((2**i)) thread(s)"
    julia --project=. --threads=$((2**i)) ParallelMergeCSR-Benchmark.jl
done