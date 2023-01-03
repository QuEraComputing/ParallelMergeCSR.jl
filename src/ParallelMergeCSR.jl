module ParallelMergeCSR

using SparseArrays
using SparseArrays: AbstractSparseMatrixCSC,
                    DenseInputVecOrMat,
                    getcolptr,
                    getrowval,
                    getnzval

using LinearAlgebra: Adjoint,adjoint,Transpose,transpose

using Base.Threads

using Atomix

mutable struct Coordinate
    x::Int
    y::Int
end

# Want to give the illusion that we're still using 0-based coordinates when the actual
# values themselves are 1-based
# 
# diagonal -> i + j = k 
# a_len and b_len can stay the same from the paper
# 
# a -> row-end offsets so really pass in a[2:end]
# b -> "natural" numbers
function merge_path_search(diagonal::Int, a_len::Int, b_len::Int, a, b)
    # Diagonal search range (in x coordinate space)
    x_min = max(diagonal - b_len, 0)
    x_max = min(diagonal, a_len)
    # 2D binary-search along diagonal search range
    while (x_min < x_max)
        pivot = (x_min + x_max) >> 1
        if (a[pivot + 1] <= b[diagonal - pivot])
            x_min = pivot + 1
        else
            x_max = pivot
        end
    end

    return Coordinate(
        min(x_min, a_len),
        diagonal - x_min
    )

end

#=
Algorithm originally designed for CSR now needs to work for CSC

CSC has fields:
* m - number of rows
* n - number of columns
* colptr::Vector - column j is in colptr[j]:(colptr[j+1]-1)
* rowval::Vector - row indices of stored values
* nzval::Vector  - stored values, typically nonzeros

CSR used to have fields: 
* m - number of rows
* n - number of columns
* rowptr::Vector - row i is in rowptr[i]:(rowptr[i+1]-1)
* colval::Vector - col indices of stored values
* nzval::Vector - Stored values, typically non-zeros

Given a matrix A in CSC, if you do transpose(A), indexing
individual elements is fine but it's a lazy transpose.
To materialize the changes, probably requires you perform some copy operation
e.g. can see changes via dump(copy(transpose(m))) where m isa AbstractMatrixCSC

* NOTE: we don't want the transpose to actually change the matrix dimensions,
we do it just so the INTERNAL representation of the matrix looks like a CSR
=#


# TODO: 
# 1. add @inbounds to remove bounds check
# 2. add update tests to work with CSC matrices. you can use sprand(...) to generate some matrix
# StridedVector is too restrictive, not even sure how you end up with a StridedVector in the first place 
# Axβ = y
function merge_csr_mv!(α::Number,A::AbstractSparseMatrixCSC, input::StridedVector, output::StridedVector, op)

    # transpose the CSC to CSR
    ## colptr in CSC equiv. to rowptr in CSR
    ## rowval in CSC equiv. to colval in CSR
    ## rows are now columns so the m x n dimensions after transpose are n x m

    # At = copy(transpose(A))
    # row_end_offsets = At.colptr[2:end]

    nzv = getnzval(A)
    rv = getrowval(A)
    cp = getcolptr(A)

    nnz = length(nzv) 
    
    # nrows = length(cp) - 1 can give the wrong number of rows!
    nrows = A.m

    nz_indices = rv
    row_end_offsets = cp[2:end] # nzval ordering is diff for diff formats
    num_merge_items = length(nzv) + nrows # preserve the dimensions of the original matrix

    num_threads = nthreads()
    items_per_thread = (num_merge_items + num_threads - 1) ÷ num_threads

    row_carry_out = zeros(eltype(cp), num_threads)
    value_carry_out = zeros(eltype(output), num_threads) # value must match output

    # Julia threads start id by 1, so make sure to offset!
    @threads for tid in 1:num_threads
        diagonal = min(items_per_thread * (tid - 1), num_merge_items)
        diagonal_end = min(diagonal + items_per_thread, num_merge_items)

        # Get starting and ending thread coordinates (row, nzv)
        thread_coord = merge_path_search(diagonal, nrows, nnz, row_end_offsets, nz_indices)
        thread_coord_end = merge_path_search(diagonal_end, nrows, nnz, row_end_offsets, nz_indices)

        # Consume merge items, whole rows first
        running_total = zero(eltype(output))
        while thread_coord.x < thread_coord_end.x
            while thread_coord.y < row_end_offsets[thread_coord.x + 1] - 1
                @inbounds running_total += op(nzv[thread_coord.y + 1]) * input[rv[thread_coord.y + 1]]
                thread_coord.y += 1
            end

            @inbounds output[thread_coord.x + 1] += α * running_total
            running_total = zero(eltype(output))
            thread_coord.x += 1 
        end
       
        # May have thread end up partially consuming a row.
        # Save result form partial consumption and do one pass at the end to add it back to y
        while thread_coord.y < thread_coord_end.y
            @inbounds running_total += op(nzv[thread_coord.y + 1]) * input[rv[thread_coord.y + 1]] 
            thread_coord.y += 1
        end

        # Save carry-outs
        row_carry_out[tid] = thread_coord_end.x + 1
        value_carry_out[tid] = running_total

    end

    for tid in 1:num_threads
        @inbounds if row_carry_out[tid] <= nrows
            @inbounds output[row_carry_out[tid]] += α * value_carry_out[tid]
        end
    end

end


# C = adjoint(A)Bα + Cβ
# C = transpose(A)B + Cβ
# C = xABα + Cβ
for (T, t) in ((Adjoint, adjoint), (Transpose, transpose))
    @eval function SparseArrays.mul!(C::StridedVecOrMat, xA::$T{<:Any,<:AbstractSparseMatrixCSC}, B::DenseInputVecOrMat, α::Number, β::Number)
        # obtains the original matrix underneath the "lazy wrapper"
        A = xA.parent
        size(A, 2) == size(C, 1) || throw(DimensionMismatch())
        size(A, 1) == size(B, 1) || throw(DimensionMismatch())
        size(B, 2) == size(C, 2) || throw(DimensionMismatch())
        if β != 1
            # assume the rmul! intended to come from SparseArrays
            ## (could implement a threaded version of that as well, maybe piggyback off of impl branch)
            # preemptively handle β so we just handle C = ABα + C
            β != 0 ? SparseArrays.rmul!(C, β) : fill!(C, zero(eltype(C)))
        end
        # move multiplication by alpha into the multithreaded part
        for (col_idx, input) in enumerate(eachcol(B))
            # merge_csr_mv!(A, x, β, y, op)
            output = @view C[:, col_idx]
            merge_csr_mv!(α, A, input, output, $t)
        end
        C
        # end of @eval macro
    end
    # end of for loop
end


## function names should be suffixed w/ exclamation mark
function atomic_add!(a::AbstractVector{T},b::T) where T <: Real
    # There's an atomic_add! in Threads but has signature (x::Atomic{T}, val::T) where T <: ArithmeticTypes
    # you'd have to wrap each element of the array with Atomic
    # Objective: for each element in AbstractVector, atomically add b to the element
    # Can also use Atomix.jl
    for idx in 1:length(a)
        Atomix.@atomic a[idx] += b
    end
    
end

function atomic_add!(a::AbstractVector{T},b::T) where T <: Complex
    # return type representing real part
    view_a = reinterpret(reshape, real(T),a)
    real_b = real(b)
    imag_b = imag(b)

    # mutating the view also mutates the underlying array it came from
    for idx in 1:size(view_a, 2)
        Atomix.@atomic view_a[1, idx] += real_b
        Atomix.@atomic view_a[2, idx] += imag_b
    end
end

function SparseArrays.mul!(C::StridedVecOrMat, A::AbstractSparseMatrixCSC, B::DenseInputVecOrMat, α::Number, β::Number)
    size(A, 2) == size(B, 1) || throw(DimensionMismatch())
    size(A, 1) == size(C, 1) || throw(DimensionMismatch())
    size(B, 2) == size(C, 2) || throw(DimensionMismatch())
    nzv = nonzeros(A)
    rv = rowvals(A)
    if β != 1
        # assume the rmul! intended to come from SparseArrays
        ## (could implement a threaded version of that as well, maybe piggyback off of impl branch)
        # preemptively handle β so we just handle C = ABα + C
        β != 0 ? SparseArrays.rmul!(C, β) : fill!(C, zero(eltype(C)))
    end
    # iterate over the columns of C
    for k in 1:size(C, 2)
        # iterate over columns of A
        @threads for col in 1:size(A, 2)
            # multiply each element of B times alpha (single value)
            @inbounds αxj = B[col,k] * α
            for j in nzrange(A, col) # range of indices in nzv of A restricted to column
                @inbounds val = nzv[j]*αxj
                @inbounds row = rv[j]
                @inbounds out = @view C[row:row, k]
                atomic_add!(out,val) 
            end
        end
    end
    C
end

# end of the module
end