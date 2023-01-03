using Test
using ParallelMergeCSR: merge_csr_mv!
using SparseArrays

## NOTE: Sparse matrices are converted to dense form in the @test's
##       considering that our redefinition of SparseArrays.mul! seems to
##       interfere

## New function signature for merge_csr_mv!, it is now
## merge_csr_mv!(α, A, input, output, op)
## so should be A*input*α = output with some op
@testset "Extreme Cases" begin

    @testset "Singleton" begin
        A = sparse(reshape([1], 1, 1))

        x = rand(1)

        y = zeros(size(A, 1))

        merge_csr_mv!(0.3, A, x, y, transpose)

        @test Matrix(A) * x * 0.3 == y
    end

    @testset "Single row" begin

        # 10 x 1 converted to 1 x 10
        A = SparseArrays.sprand(10, 1, 0.3)

        # x needs to be 10 x 1 
        x = rand(size(A, 1))

        y = zeros(eltype(x), 1)

        merge_csr_mv!(1.1, A, x, y, transpose)

        @test (transpose(Matrix(A)) * x) * 1.1 ≈ y
    end

    @testset "Single column" begin

        # 1 x 10 is now 10 x 1
        A = SparseArrays.sprand(1, 10, 0.3)
        
        # needs to be 1 x 1
        x = rand(1)

        # 10 x 1
        y = zeros(size(A,2))
        
        merge_csr_mv!(0.7, A, x, y, adjoint)

        y_exact = 0.7 * adjoint(Matrix(A)) * x

        @test y_exact ≈ y
    end
end

@testset "Square" begin
    # 10 x 10 with 30% chance of entry being made
    A = SparseArrays.sprand(10,10,0.3)

    # 10 x 1
    x = rand(10)

    y = zeros(size(x))

    merge_csr_mv!(1.1, A, x, y, adjoint)

    @test (adjoint(Matrix(A)) * x) * 1.1 ≈ y

end

@testset "4x6" begin
    # create matrix
    m = [10 20 0 0 0 0;
         0 30 0 40 0 0;
         0 0 50 60 70 0;
         0 0 0 0 0 80]

    # get into CSC form
    A = sparse(m)

    # create vector
    # x = [5,2,3,1,8,2]
    x = [5,2,3,1]

    # create empty solution
    y = zeros(Int64, size(A, 2))
    
    # multiply
    merge_csr_mv!(2.0, A, x, y, adjoint)


    @test transpose(m) * x * 2.0 == y
end

@testset "100x100" begin
    # create matrix
    A = SparseArrays.sprand(100, 100, 0.3)

    # create vector
    x = rand(1:100, 100, 1)

    # create empty solution
    y = zeros(size(A, 1))

    merge_csr_mv!(3, A, x, y, transpose)

    @test Matrix(A) * x * 3 ≈ y
end

@testset "Matrix x Matrix" begin

    ## Calculate AXα
    # Create Matrices
    A = SparseArrays.sparse(rand(1:10, 3, 2))
    X = rand(1:10, 2, 4)

    # set alpha to 1.0 for now
    α = 1.0

    # Create place to store solution
    Y = zeros(3, 4)

    # iterate
    for (idx, col) in enumerate(eachcol(X))
        # merge_csr_mv!(A, x, β, y, op)
        Y_view = @view Y[:, idx]
        merge_csr_mv!(α, A, col, Y_view, x -> x)
        # merge_csr_mv!(A, col, α, Y_view, x -> x)
    end

    @test Matrix(A) * X == Y 

end