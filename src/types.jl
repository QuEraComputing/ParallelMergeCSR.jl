abstract type AbstractSparseMatrixCSR{Tv,Ti} <: AbstractSparseMatrix{Tv,Ti} end

struct SparseMatrixCSR{Tv,Ti} <: AbstractSparseMatrixCSR{Tv,Ti} 
    m::Int                  # Number of rows
    n::Int                  # Number of columns
    rowptr::Vector{Ti}      # Row i is in rowptr[i]:(rowptr[i+1]-1)
    colval::Vector{Ti}      # Col indices of stored values
    nzval::Vector{Tv}      # Stored values, typically nonzeros
end



Base.size(sm::AbstractSparseMatrixCSR) = (sm.m, sm.n)

function Base.getindex(sm::AbstractSparseMatrixCSR{Tv,Ti},row::Int,col::Int) where {Tv,Ti}
    @assert 0 < row ≤ size(sm,1)
    @assert 0 < col ≤ size(sm,2)

    lb = sm.rowptr[row]
    ub = sm.rowptr[row+1]
    ind = searchsortedlast(sm.colval[lb:ub],col)
    
    sm.colval[ind] == col && return sm.nzval[ind]

    return zero(Tv)

end


# TODO:
# create constructor from SparseMatrixCSC
# create constructor from Dense Matrix
# add test for canonical form for CSR
# add method to remove zero values
# add method to sort column indices