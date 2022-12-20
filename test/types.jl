using Test

using ParallelMergeCSR
using SparseArrays
using LinearAlgebra

@testset "Interface" begin
    # from Wikipedia
    n = 4
    m = 6
    nzval  = [ 10, 20, 30, 40, 50, 60, 70, 80 ]
    colval = [  1,  2,  2,  4,  3,  4,  5,  6 ]   
    rowptr = [  1,  3,  5,  8,  9 ]
    
    a = SparseMatrixCSR(n, m, rowptr, colval, nzval)
    
    @test size(a) == (n,m)
    @test indptr(a) == rowptr
    @test colvals(a) == colval
    @test nonzeros(a) == nzval

    @test Matrix(a) == [
        [10  20   0   0   0   0]
        [0  30   0  40   0   0]
        [0   0  50  60  70   0]
        [0   0   0   0   0  80]
    ]

    b = SparseMatrixCSC([
        [10  20   0   0   0   0]
        [0  30   0  40   0   0]
        [0   0  50  60  70   0]
        [0   0   0   0   0  80]]
    )

    b_csr = SparseMatrixCSR(b)

    @test size(a) == size(b_csr)
    @test indptr(a) == indptr(b_csr)
    @test colvals(a) == colvals(b_csr)
    @test nonzeros(a) == nonzeros(b_csr)
    @test a == b_csr


    @test a == SparseMatrixCSR([
        [10  20   0   0   0   0]
        [0  30   0  40   0   0]
        [0   0  50  60  70   0]
        [0   0   0   0   0  80]]
    )

    @test b[1,1] == 10
    @test b[2,2] == 30
    @test b[4,1] == 0
    
end
