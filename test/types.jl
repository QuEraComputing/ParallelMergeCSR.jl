using ParallelMergeCSR
using SparseArrays

a = sprand(100, 100, 0.01)
b = SparseMatrixCSR(a.n, a.m, a.colptr, a.rowval, a.nzval)




