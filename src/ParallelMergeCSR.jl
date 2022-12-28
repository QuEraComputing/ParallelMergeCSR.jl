module ParallelMergeCSR

using SparseArrays: AbstractSparseMatrixCSC
using LinearAlgebra:Adjoint,adjoint,Transpose,transpose

using Base.Threads

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

# Ax = y
# A matrix
# x, y vectors

# TODO: overload SparseArrays.mul!
# TODO: add β and α arguments
# TODO: change SparseMatrixCSR => Transpose{AbstractSparseMatrixCSC}
#     1. colval => rowval 
#     2. rowptr => colptr
#     3. use iterface of SparseArrays to get these quantities
function merge_csr_mv!(A::AbstractSparseMatrixCSC, input::StridedVector, output::StridedVector,op)
    rv = rowvals(A)
    nzval = nonzeros(A)
    nrow = size(A,2) # view CSC tranpose as CSR.

    row_end_offsets = rowvals[2:end]
    nnz_indices = collect(1:length(nzval))
    num_merge_items = length(nzval) + nrow

    num_threads = nthreads()
    items_per_thread = (num_merge_items + num_threads - 1) ÷ num_threads

    row_carry_out = zeros(Int, num_threads)
    value_carry_out = zeros(Float64, num_threads)

    # Julia threads start id by 1, so make sure to offset!
    @threads for tid in 1:num_threads
        diagonal = min(items_per_thread * (tid - 1), num_merge_items)
        diagonal_end = min(diagonal + items_per_thread, num_merge_items)

        # Get starting and ending thread coordinates (row, nnz)
        thread_coord = merge_path_search(diagonal, nrow, length(nzval), row_end_offsets, nnz_indices)
        thread_coord_end = merge_path_search(diagonal_end, nrow, length(nzval), row_end_offsets, nnz_indices)

        # Consume merge items, whole rows first
        running_total = zero(eltype(y)) 
        while thread_coord.x < thread_coord_end.x
            while thread_coord.y < row_end_offsets[thread_coord.x + 1] - 1
                running_total += op(nzval[thread_coord.y + 1]) * input[rv[thread_coord.y + 1]]
                thread_coord.y += 1
            end

            output[thread_coord.x + 1] = running_total
            running_total = zero(eltype(y)) 
            thread_coord.x += 1 
        end
       
        # May have thread end up partially consuming a row.
        # Save result form partial consumption and do one pass at the end to add it back to y
        while thread_coord.y < thread_coord_end.y
            running_total += op(nzval[thread_coord.y + 1]) * input[rv[thread_coord.y + 1]]
            thread_coord.y += 1
        end

        # Save carry-outs
        row_carry_out[tid] = thread_coord_end.x + 1
        value_carry_out[tid] = running_total

    end

    # Diagonistcs
    # println("row_carry_out: $row_carry_out")
    # println("value_carry_out: $value_carry_out")
    # println("Incomplete y: $y")
    for tid in 1:num_threads
        if row_carry_out[tid] <= nrow
            output[row_carry_out[tid]] += value_carry_out[tid]
        end
    end

end


# y => C[:,k]
# x => B[:,k]
for (T, t) in ((Adjoint, adjoint), (Transpose, transpose))
    @eval function SparseArrays.mul!(C::StridedVecOrMat, xA::$T{<:Any,<:AbstractSparseMatrixCSC}, B::DenseInputVecOrMat, α::Number, β::Number)
        A = xA.parent
        size(A, 2) == size(C, 1) || throw(DimensionMismatch())
        size(A, 1) == size(B, 1) || throw(DimensionMismatch())
        size(B, 2) == size(C, 2) || throw(DimensionMismatch())
        # nzv = nonzeros(A)
        # rv = rowvals(A)
        if β != 1
            β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
        end
        for k in 1:size(C, 2)
            # parallel implementation goes here acting on slice C[:,k]
            y = @view B[:,k]
            x = @view C[:,k]
            merge_csr_mv!(A, x, y, $t)
        end
        C
    end
end


function atomic_add(a::AbstractVector{T},b::T) where T <: Real
    @atomic a[1] += b
end

function atomic_add(a::AbstractVector{T},b::T) where T <: Complex
    view_a = reinterpret(real(T),a)
    real_b = real(b)
    imag_b = imag(b)

    @atomic view_a[1] += real_b
    @atomic view_a[2] += imag_b
end

function SparseArrays.mul!(C::StridedVecOrMat, A::AbstractSparseMatrixCSC, B::DenseInputVecOrMat, α::Number, β::Number)
    size(A, 2) == size(B, 1) || throw(DimensionMismatch())
    size(A, 1) == size(C, 1) || throw(DimensionMismatch())
    size(B, 2) == size(C, 2) || throw(DimensionMismatch())
    nzv = nonzeros(A)
    rv = rowvals(A)
    if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
    end
    for k in 1:size(C, 2)
        @threads for col in 1:size(A, 2)
            @inbounds αxj = B[col,k] * α
            for j in nzrange(A, col)
                @inbounds val = nzv[j]*αxj
                @inbounds row = rv[j]
                @inbounds out = C[row:row, k] # pass element as view
                atomic_add(out,val) 
            end
        end
    end
    C
end


end