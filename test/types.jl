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
    
    b = SparseMatrixCSR(n, m, rowptr, colval, nzval)
    
    @test size(b) == (n,m)
    @test indptr(b) == rowptr
    @test colvals(b) == colval
    @test nonzeros(b) == nzval

    @test Matrix(b) == [
        [10  20   0   0   0   0]
        [0  30   0  40   0   0]
        [0   0  50  60  70   0]
        [0   0   0   0   0  80]
    ]

    a = SparseMatrixCSC([
        [10  20   0   0   0   0]
        [0  30   0  40   0   0]
        [0   0  50  60  70   0]
        [0   0   0   0   0  80]]
    )

    @test b ==  SparseMatrixCSR(a)

    @test b[1,1] == 10
    @test b[2,2] == 30
    @test b[4,1] == 0
    
end







# tests
# ---------------
# Base.size
# Base.getindex





