using ParallelMergeCSR
using BenchmarkTools
using SparseArrays
using Random
using Test



@testset "threaded_fill!" begin
    A = zeros(Int,3,3)
    ParallelMergeCSR.threaded_fill!(A,3)

    @test A == [
        [3 3 3]
        [3 3 3]
        [3 3 3]
    ]

end

@testset "threaded_rmul!" begin
    A = ones(Int,3,3)
    ParallelMergeCSR.threaded_rmul!(A,3)

    @test A == [
        [3 3 3]
        [3 3 3]
        [3 3 3]
    ]

end


@testset "threaded_lmul!" begin
    A = ones(Int,3,3)
    ParallelMergeCSR.threaded_lmul!(A,3)

    @test A == [
        [3 3 3]
        [3 3 3]
        [3 3 3]
    ]

end


m = SparseMatrixCSR([
    [0 1 0]
    [0 1.0im 4]
    [2 0 0]
])
m_T = transpose(m)

B = ones(3)
C = zeros(promote_type(eltype(B),eltype(m_T)),3)


mul!(C,m_T,B,1,0)

