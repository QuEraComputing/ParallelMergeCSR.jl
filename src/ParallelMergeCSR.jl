module ParallelMergeCSR

using Atomix
using Base.Threads
using SparseArrays
using LinearAlgebra: Adjoint,adjoint,Transpose,transpose
using SparseArrays: AbstractSparseMatrixCSC,
                    DenseInputVecOrMat,
                    getcolptr


include("csrmv_merge.jl")
include("parallel_csc_mv.jl")

end # end of the module
