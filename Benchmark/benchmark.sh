
echo "Creating Benchmarks CSV"
julia --project=. prepare_csv.jl

echo "Benchmarking SparseArrays"
julia --project=. SparseArrays-Benchmark.jl

for num_threads in {1,2,4,8,16,32,64,128}
do
    echo "Benchmarking against $num_threads thread(s)"
    julia --project=. --threads=$num_threads ParallelMergeCSR-Benchmark.jl
done