module ParallelMergeCSR

using Atomix
using Base.Threads
using SparseArrays
using LinearAlgebra: Adjoint,adjoint,Transpose,transpose
using SparseArrays: AbstractSparseMatrixCSC,
                    DenseInputVecOrMat,
                    getcolptr


export mul!

include("parallel_csr_mv.jl")

end # end of the module
