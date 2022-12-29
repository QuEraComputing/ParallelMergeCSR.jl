using Test
using ParallelMergeCSR
using SparseArrays

## NOTE: Sparse matrices are converted to dense form in the @test's
##       considering that our redefinition of SparseArrays.mul! seems to
##       interfere

@testset "Extreme Cases" begin

    @testset "Singleton" begin
        A = sparse(reshape([1], 1, 1))

        x = rand(1)

        y = zeros(size(A, 1))

        ParallelMergeCSR.merge_csr_mv!(A, x, 1.0, y, transpose)

        @test Matrix(A) * x == y
    end

    @testset "Single row" begin

        A = SparseArrays.sprand(10, 1, 0.3)

        x = rand(1:10, 1)

        y = zeros(size(A, 1))

        ParallelMergeCSR.merge_csr_mv!(A, x, 1.0, y, adjoint)

        @test Matrix(A) * x ≈ y
    end

    @testset "Single column" begin

        A = SparseArrays.sprand(10, 1, 0.3)

        x = rand(1:10, 1)

        y = zeros(size(A, 1))

        ParallelMergeCSR.merge_csr_mv!(A, x, 1.0, y, transpose)

        @test Matrix(A) * x ≈ y
    end
end

@testset "Square" begin
    # 10 x 10 with 30% chance of entry being made
    A = SparseArrays.sprand(10,10,0.3)

    x = rand(10)

    y = zeros(size(A, 1))

    ParallelMergeCSR.merge_csr_mv!(A, x, 1.1, y, adjoint)

    @test (Matrix(A) * x) * 1.1 ≈ y

end

@testset "4x6" begin
    # create matrix
    m = [10 20 0 0 0 0;
         0 30 0 40 0 0;
         0 0 50 60 70 0;
         0 0 0 0 0 80]

    # get into CSR form
    A = sparse(m)

    # create vector
    x = [5,2,3,1,8,2]

    # create empty solution
    y = zeros(Int64, size(A, 1))
    
    # multiply
    ParallelMergeCSR.merge_csr_mv!(A, x, 2.0, y, adjoint)

    @test m * x * 2.0 == y
end

@testset "100x100" begin
    # create matrix
    A = SparseArrays.sprand(100, 100, 0.3)

    # create vector
    x = rand(1:100, 100, 1)

    # create empty solution
    y = zeros(size(A, 1))

    ParallelMergeCSR.merge_csr_mv!(A, x, 3, y, transpose)

    @test Matrix(A) * x * 3 ≈ y
end