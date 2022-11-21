module ParallelMergeCSR

using SparseArrays: SparseArrays, 
                    AbstractSparseMatrix, 
                    DenseInputVecOrMat, 
                    nonzeros, 
                    nonzeroinds, 
                    rowvals, 
                    nzrange
using LinearAlgebra

export SparseMatrixCSR

include("types.jl")
include("threaded.jl")

# include("gpu.jl")
# include("distributed.jl")


end