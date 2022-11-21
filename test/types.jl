
abstract type AbstractSparseMatrixCSR{Tv,Ti} <: AbstractSparseMatrix{Tv,Ti} end

struct SparseMatrixCSR{Tv,Ti} <: AbstractSparseMatrixCSR{Tv,Ti} 
    m::Int                  # Number of rows
    n::Int                  # Number of columns
    rowptr::Vector{Ti}      # Row i is in rowptr[i]:(rowptr[i+1]-1)
    colval::Vector{Ti}      # Col indices of stored values
    nzval::Vector{Tv}       # Stored values, typically nonzeros
end
