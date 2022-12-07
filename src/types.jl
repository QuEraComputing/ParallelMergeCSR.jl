abstract type AbstractSparseMatrixCSR{Tv,Ti} <: AbstractSparseMatrix{Tv,Ti} end

struct SparseMatrixCSR{Tv,Ti} <: AbstractSparseMatrixCSR{Tv,Ti} 
    m::Int                  # Number of rows
    n::Int                  # Number of columns
    rowptr::Vector{Ti}      # Row i is in rowptr[i]:(rowptr[i+1]-1)
    colval::Vector{Ti}      # Col indices of stored values
    nzval::Vector{Tv}      # Stored values, typically nonzeros
end



function SparseMatrixCSR(sm::AbstractSparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    n,m = size(sm)

    in_colptr = indptr(sm)
    in_rowval = rowvals(sm)
    in_nzval = nonzeros(sm)

    nnz = in_colptr[end]

    rowptr = zeros(eltype(in_colptr),n+1)
    colval = similar(in_rowval)
    nzval  = similar(in_nzval)

    @inbounds for rowval in in_rowval
        rowptr[rowval] += 1
    end

    cumsum = one(Ti)
    @inbounds for row in one(Ti):n
        temp = rowptr[row]
        rowptr[row] = cumsum
        cumsum += temp
    end
    @inbounds rowptr[n+1] = nnz

    for col in 1:m
        @inbounds for ind in nzrange(sm,col)
            row = in_rowval[ind]
            dest = rowptr[row]

            colval[dest] = col
            nzval[dest] = in_nzval[ind]

            rowptr[row] += 1
        end
    end
    
    last = one(Ti)
    @inbounds for row in 1:n
        temp = rowptr[row]
        rowptr[row] = last
        last = temp
    end

    return SparseMatrixCSR{Tv,Ti}(n,m,rowptr,colval,nzval)
end

getrowptr(sm::SparseMatrixCSR) = getfield(sm,:rowptr)
getcolval(sm::SparseMatrixCSR) = getfield(sm,:colval)
getnzval(sm::SparseMatrixCSR)  = getfield(sm,:nzval)



# abstract interface
indptr(sm::AbstractSparseMatrixCSR) = getrowptr(sm)
colvals(sm::AbstractSparseMatrixCSR) = getcolval(sm)
SparseArrays.nonzeros(sm::AbstractSparseMatrixCSR) = getnzval(sm)

indptr(sm::AbstractSparseMatrixCSC) = getcolptr(sm)
SparseArrays.nzrange(sm::AbstractSparseMatrixCSR, row::Integer) = getrowptr(sm)[row]:(getrowptr(sm)[row+1]-1)



Base.size(sm::AbstractSparseMatrixCSR) = (sm.m, sm.n)

function Base.getindex(sm::AbstractSparseMatrixCSR{Tv,Ti},row::Int,col::Int) where {Tv,Ti}
    @assert 0 < row ≤ size(sm,1)
    @assert 0 < col ≤ size(sm,2)

    colval = colvals(sm)
    nzval  = nonzeros(sm)

    nzr = nzrange(sm,row)

    row_colval = colval[nzr]
    row_nzval = nzval[nzr]
    ind = searchsortedfirst(row_colval,col)

    ind ≤ length(row_colval) && row_colval[ind] == col && return row_nzval[ind]

    return zero(Tv)

end

function LinearAlgebra.Matrix(sm::AbstractSparseMatrixCSR{Tv}) where Tv
    dm = zeros(Tv,size(sm))
    nrow = size(sm,1)

    rowptr = indptr(sm)
    colval = colvals(sm)
    nzval  = nonzeros(sm)

    for row in 1:nrow
        lb = rowptr[row]
        ub = rowptr[row+1]-1
        for ind in lb:ub
            dm[row,colval[ind]] = nzval[ind]
        end
    end

    return dm 

end

# TODO:
# create constructor from SparseMatrixCSC
# create constructor from Dense Matrix
# add test for canonical form for CSR
# add method to remove zero values
# add method to sort column indices