using ParallelMergeCSR
using BenchmarkTools
using SparseArrays

n = 100000
A = sprand(n,n,log(n)/n)
x = rand(n)
y = rand(n)


bm_1 = @benchmark ParallelMergeCSR.threaded_mul_1!(y,A,x,1,1)
show(stdout,bm_1)
