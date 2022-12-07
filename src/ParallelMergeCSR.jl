module ParallelMergeCSR

using SparseArrays: SparseArrays, 
                    AbstractSparseMatrix,
                    AbstractSparseMatrixCSC, 
                    DenseInputVecOrMat, 
                    nonzeros, 
                    nonzeroinds, 
                    rowvals, 
                    getcolptr,
                    nzrange
using LinearAlgebra

export SparseMatrixCSR,
    Matrix,
    indptr,
    colvals,
    getnzval

include("types.jl")
include("threaded.jl")

# include("gpu.jl")
# include("distributed.jl")


end