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

export SparseMatrixCSR

include("types.jl")
include("threaded.jl")

# include("gpu.jl")
# include("distributed.jl")


end