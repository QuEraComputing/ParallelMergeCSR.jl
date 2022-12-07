using ParallelMergeCSR
using BenchmarkTools
using SparseArrays
using Random



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




