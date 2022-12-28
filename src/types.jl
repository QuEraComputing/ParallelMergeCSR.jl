abstract type AbstractSparseMatrixCSR{Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti} end

struct SparseMatrixCSR{Tv,Ti} <: AbstractSparseMatrixCSR{Tv,Ti} 
    m::Int                  # Number of rows
    n::Int                  # Number of columns
    rowptr::Vector{Ti}      # Row i is in rowptr[i]:(rowptr[i+1]-1)
    colval::Vector{Ti}      # Col indices of stored values
    nzval::Vector{Tv}      # Stored values, typically nonzeros
    function SparseMatrixCSR{Tv,Ti}(m::Integer, n::Integer, rowptr::Vector{Ti},
                            colval::Vector{Ti}, nzval::Vector{Tv}) where {Tv,Ti<:Integer}
        @noinline throwsz(str, lbl, k) =
            throw(ArgumentError("number of $str ($lbl) must be ≥ 0, got $k"))
        m < 0 && throwsz("rows", 'm', m)
        n < 0 && throwsz("columns", 'n', n)
        new{Tv,Ti}(Int(m), Int(n), rowptr, colval, nzval)
    end
end
function SparseMatrixCSR(m::Integer, n::Integer, rowptr::Vector, colval::Vector, nzval::Vector)
    Tv = eltype(nzval)
    Ti = promote_type(eltype(rowptr), eltype(colval))
    SparseArrays.sparse_check_Ti(m, n, Ti)
    sparse_check(m, rowptr, colval, nzval)
    # silently shorten colval and nzval to usable index positions.
    maxlen = abs(widemul(m, n))
    isbitstype(Ti) && (maxlen = min(maxlen, typemax(Ti) - 1))
    length(colval) > maxlen && resize!(colval, maxlen)
    length(nzval) > maxlen && resize!(nzval, maxlen)
    SparseMatrixCSR{Tv,Ti}(m, n, rowptr, colval, nzval)
end

function sparse_check(m::Integer, rowptr::Vector{Ti}, colval, nzval) where Ti
    # String interpolation is a performance bottleneck when it's part of the same function,
    # ensure we only do it once committed to the error.
    throwstart(ckp) = throw(ArgumentError("$ckp == rowptr[1] != 1"))
    throwmonotonic(ckp, ck, k) = throw(ArgumentError("$ckp == rowptr[$(k-1)] > rowptr[$k] == $ck"))

    sparse_check_length("rowptr", rowptr, m+1, String) # don't check upper bound
    ckp = Ti(1)
    ckp == rowptr[1] || throwstart(ckp)
    @inbounds for k = 2:m+1
        ck = rowptr[k]
        ckp <= ck || throwmonotonic(ckp, ck, k)
        ckp = ck
    end
    sparse_check_length("colval", colval, ckp-1, Ti)
    sparse_check_length("nzval", nzval, 0, Ti) # we allow empty nzval !!!
end
function sparse_check_length(rowstr, colval, minlen, Ti)
    throwmin(len, minlen, rowstr) = throw(ArgumentError("$len == length($rowstr) < $minlen"))
    throwmax(len, max, rowstr) = throw(ArgumentError("$len == length($rowstr) >= $max"))

    len = length(colval)
    len >= minlen || throwmin(len, minlen, rowstr)
    !isbitstype(Ti) || len < typemax(Ti) || throwmax(len, typemax(Ti), rowstr)
end

function SparseMatrixCSR(sm::AbstractSparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    n,m = size(sm)

    in_rowptr = indptr(sm)
    in_colval = rowvals(sm)
    in_nzval = nonzeros(sm)

    nnz = in_rowptr[end]

    rowptr = zeros(eltype(in_rowptr),n+1)
    colval = similar(in_colval)
    nzval  = similar(in_nzval)

    @inbounds for colval in in_colval
        rowptr[colval] += 1
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
            row = in_colval[ind]
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

function SparseMatrixCSR(dm::AbstractMatrix{Tv}) where Tv
    n,m = size(dm)

    rowptr = zeros(Int,n+1)
    nz = 1
    for row in 1:n
        rowptr[row] = nz
        row_view = @view dm[row,:]
        nz += count(!iszero, row_view)
    end
    rowptr[end] = nz

    colval = zeros(Int,nz-1)
    nzval  = zeros(Tv ,nz-1)
    for row in 1:n
        ind = rowptr[row]
        row_view = @view dm[row,:]
        for (i,v) in enumerate(row_view)
            if !iszero(v)
                colval[ind] = i
                nzval[ind] = v
                ind += 1
            end
        end
    end

    return SparseMatrixCSR(n,m,rowptr,colval,nzval)
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
# add method to remove zero values
# add method to sort column indices