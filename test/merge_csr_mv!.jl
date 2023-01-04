using Test
using ParallelMergeCSR: merge_csr_mv!
using SparseArrays

## NOTE: Sparse matrices are converted to dense form in the @test's
##       considering that our redefinition of SparseArrays.mul! seems to
##       interfere
@testset "Extreme Cases" begin

    @testset "Singleton (Real)" begin
        A = sparse(reshape([1], 1, 1))

        x = rand(1)

        y = zeros(size(A, 1))

        merge_csr_mv!(0.3, A, x, y, transpose)

        @test Matrix(A) * x * 0.3 == y
    end

    @testset "Singleton (Complex)" begin
        A = sparse(reshape([1.0+2.5im], 1, 1))

        x = rand(1)

        y = zeros(eltype(A), size(A, 1))

        merge_csr_mv!(10.1, A, x, y, adjoint)

        @test adjoint(Matrix(A)) * x * 10.1 == y
    end

    @testset "Single row (Real)" begin

        # 10 x 1 converted to 1 x 10
        A = 10.0 * SparseArrays.sprand(10, 1, 0.3)

        # x needs to be 10 x 1 
        x = rand(size(A, 1))

        y = zeros(eltype(A), 1)

        merge_csr_mv!(1.1, A, x, y, transpose)

        @test (transpose(Matrix(A)) * x) * 1.1 ≈ y
    end

    @testset "Single row (Complex)" begin

        # 10 x 1 is treated as 1 x 10 inside merge_csr_mv!
        A = SparseArrays.sprand(Complex{Float64}, 10, 1, 0.3)

        x = rand(eltype(A), size(A, 1))

        y = zeros(eltype(A), 1)

        merge_csr_mv!(1.1, A, x, y, transpose)

        @test (transpose(Matrix(A)) * x) * 1.1 ≈ y
    end


    #= 
        Single Column test wouldn't work because 
        merge_csr_mv! is designed to handle Ax = y where
        A is m x n, x is n x 1, and y must be m x 1.

        Given a single column A that is 1 x z, merge_csr_mv! treats it as 
        z x 1, which means y is constrained to 1 x 1 and the correct
        output would be z x 1, which merge_csr_mv! is not designed for. 

        On the other hand, the overridden SparseArrays.mul! should work
    =#
end

# test fails, y seems to have lots of zero-entries
@testset "Square (Real)" begin
    A = SparseArrays.sprand(10,10,0.3)

    # 10 x 1
    x = rand(10)

    # 10 x 1
    y = zeros(size(x))

    merge_csr_mv!(1.1, A, x, y, adjoint)

    @test (adjoint(Matrix(A)) * x) * 1.1 ≈ y

end

@testset "Square (Complex)" begin
    A = SparseArrays.sprand(Complex{Float64}, 10, 10, 0.3)

    x = 10 * rand(Complex{Float64}, 10)

    y = zeros(eltype(A), size(x))

    merge_csr_mv!(1.1, A, x, y, adjoint)

    @test (adjoint(Matrix(A)) * x) * 1.1 ≈ y
end

@testset "4x6 (Real)" begin
    # Taken from: https://en.wikipedia.org/wiki/Sparse_matrix
    m = [10 20 0 0 0 0;
         0 30 0 40 0 0;
         0 0 50 60 70 0;
         0 0 0 0 0 80]

    # get into CSC form
    A = sparse(m)

    # create vector
    x = [5,2,3,1]

    # create empty solution
    y = zeros(Int64, size(A, 2))
    
    # multiply
    merge_csr_mv!(2.0, A, x, y, adjoint)


    @test Matrix(adjoint(m)) * x * 2.0 == y
end

@testset "4 x 6 (Complex)" begin
    # Taken from: https://en.wikipedia.org/wiki/Sparse_matrix
    m = [10 20 0 0 0 0;
    0 30 0 40 0 0;
    0 0 50 60 70 0;
    0 0 0 0 0 80]

    # get into CSC form
    A = sparse(m)

    # create vector
    x = 22.1 * rand(Complex{Float64}, 4)

    # create empty solution
    y = zeros(eltype(x), size(A, 2))

    # multiply
    merge_csr_mv!(2.0, A, x, y, adjoint)

    @test Matrix(adjoint(m)) * x * 2.0 == y

end

@testset "100x100 (Real)" begin
    # create matrix
    A = SparseArrays.sprand(100, 100, 0.3)

    # create vector
    x = rand(100)

    # create empty solution
    y = zeros(size(A, 1))

    merge_csr_mv!(3.0, A, x, y, transpose)

    @test transpose(Matrix(A)) * x * 3 ≈ y
end

@testset "100x100 (Complex)" begin
    # create matrix
    A = SparseArrays.sprand(Complex{Float64}, 100, 100, 0.3)

    # create vector
    x = rand(Complex{Float64}, 100)

    # create empty solution
    y = zeros(eltype(x), size(A, 1))

    merge_csr_mv!(3.0, A, x, y, transpose)

    @test transpose(Matrix(A)) * x * 3 ≈ y
end

#=
    NOTE: While merge_csr_mv! can be used this way, the overriden SparseArrays.mul!
    should be the preferred method to do Matrix-Matrix multiplication.
=#
@testset "Matrix-Matrix (Real)" begin

    # Create Matrices
    # 3 x 2 (treated as 2 x 3 in merge_csr_mv!)
    A = sprand(3, 2, 0.4)
    # 3 x 4
    X = rand(3, 4)

    α = 9.2

    Y = zeros(2, 4)

    for (idx, col) in enumerate(eachcol(X))
        Y_view = @view Y[:, idx]
        merge_csr_mv!(α, A, col, Y_view, transpose)
    end

    @test transpose(Matrix(A)) * X * 9.2 ≈ Y 

end

@testset "Matrix-Matrix (Real)" begin

    # Create Matrices
    # 3 x 2 (treated as 2 x 3 in merge_csr_mv!)
    A = sprand(Complex{Float64}, 3, 2, 0.4)
    # 3 x 4
    X = rand(3, 4)

    α = 9.2

    Y = zeros(eltype(A), 2, 4)

    for (idx, col) in enumerate(eachcol(X))
        Y_view = @view Y[:, idx]
        merge_csr_mv!(α, A, col, Y_view, adjoint)
    end

    @test adjoint(Matrix(A)) * X * 9.2 ≈ Y 

end