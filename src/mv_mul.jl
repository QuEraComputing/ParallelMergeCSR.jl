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
function merge_csr_mv!(A::SparseMatrixCSR, x, y)
    row_end_offsets = A.rowptr[2:end]
    nnz_indices = collect(1:length(A.nzval))
    num_merge_items = length(A.nzval) + A.m

    num_threads = nthreads()
    items_per_thread = (num_merge_items + num_threads - 1) ÷ num_threads

    row_carry_out = zeros(Int, num_threads)
    value_carry_out = zeros(Float64, num_threads)

    # Julia threads start id by 1, so make sure to offset!
    @threads for tid in 1:num_threads
        diagonal = min(items_per_thread * (tid - 1), num_merge_items)
        diagonal_end = min(diagonal + items_per_thread, num_merge_items)

        # Get starting and ending thread coordinates (row, nnz)
        thread_coord = merge_path_search(diagonal, A.m, length(A.nzval), row_end_offsets, nnz_indices)
        thread_coord_end = merge_path_search(diagonal_end, A.m, length(A.nzval), row_end_offsets, nnz_indices)

        # Consume merge items, whole rows first
        running_total = 0.0        
        while thread_coord.x < thread_coord_end.x
            while thread_coord.y < row_end_offsets[thread_coord.x + 1] - 1
                running_total += A.nzval[thread_coord.y + 1] * x[A.colval[thread_coord.y + 1]]
                thread_coord.y += 1
            end

            y[thread_coord.x + 1] = running_total
            running_total = 0.0
            thread_coord.x += 1 
        end
       
        # May have thread end up partially consuming a row.
        # Save result form partial consumption and do one pass at the end to add it back to y
        while thread_coord.y < thread_coord_end.y
            running_total += A.nzval[thread_coord.y + 1] * x[A.colval[thread_coord.y + 1]]
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
        if row_carry_out[tid] <= A.m
            y[row_carry_out[tid]] += value_carry_out[tid]
        end
    end

end