using ParallelMergeCSR
using SparseArrays
using SparseArrays: getcolptr
using ProfileView
using Random
using BenchmarkTools
using Profile
using PProf
using Base.Threads

nthreads()
rng = MersenneTwister(0)
N = 10000000
A = sprand(rng,N,N,log(N)/N)
x = rand(rng,N)
y = rand(rng,N)
ParallelMergeCSR.mul!(x,(transpose(A)),y,0.0,1.0)
SparseArrays.mul!(x,(transpose(A)),y,0.0,1.0)

@benchmark ParallelMergeCSR.mul!($x,$(transpose(A)),$y,0.0,1.0) samples=100 evals=1 seconds=172800
@benchmark SparseArrays.mul!($x,$(transpose(A)),$y,0.0,1.0) samples=100 evals=1 seconds=172800
