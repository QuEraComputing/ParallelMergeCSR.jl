module ParallelMergeCSR

using Atomix
using Base.Threads
using SparseArrays
using LinearAlgebra: Adjoint,adjoint,Transpose,transpose
using SparseArrays: AbstractSparseMatrixCSC,
                    DenseInputVecOrMat,
                    getcolptr

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

# Internally treats the matrix as if its been transposed/is an adjoint
# e.g. if you pass in a 2x3, internally it's a 3x2 and can be an adjoint if you pass in `adjoint` as the op
function merge_csr_mv!(α::Number,A::AbstractSparseMatrixCSC, input::StridedVector, output::StridedVector, op)

    nzv = nonzeros(A)
    rv = rowvals(A)
    cp = getcolptr(A)

    nnz = length(nzv) 
    
    # nrows = length(cp) - 1 can give the wrong number of rows!
    nrows = A.n

    nz_indices = collect(1:nnz)
    row_end_offsets = cp[2:end] # nzval ordering is diff for diff formats
    num_merge_items = nnz + nrows # preserve the dimensions of the original matrix

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
            @inbounds while thread_coord.y < row_end_offsets[thread_coord.x + 1] - 1
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
        @inbounds row_carry_out[tid] = thread_coord_end.x + 1
        @inbounds value_carry_out[tid] = running_total

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
            # preemptively handle β so we just handle C = ABα + C
            β != 0 ? SparseArrays.rmul!(C, β) : fill!(C, zero(eltype(C)))
        end
        for (col_idx, input) in enumerate(eachcol(B))
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
        # preemptively handle β so we just handle C = ABα + C
        β != 0 ? SparseArrays.rmul!(C, β) : fill!(C, zero(eltype(C)))
    end
    for k in 1:size(C, 2)
        @threads for col in 1:size(A, 2)
            @inbounds αxj = B[col,k] * α
            for j in nzrange(A, col)
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