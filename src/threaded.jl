


function get_chunks(nnz::Integer, ndevs::Integer)
    l, r = divrem(nnz, ndevs)

    chunk_sizes =  [k > r ? l : l+1 for k in 1:ndevs]
    return vcat([1],cumsum(chunk_sizes))
end


function threaded_rmul!(A::StridedArray,α::Number)
    A_dims = size(A)
    A = vec(A)
    n = length(A)
    # split up work
    nth = Threads.nthreads()
    chunks = get_chunks(n,nth)
    # do work 
    Threads.@threads for thn in 1:nth
        @simd for i in chunks[thn]:chunks[thn+1]
            @inbounds A[i] = A[i] * α
        end
    end
    reshape(A,A_dims)
end

function threaded_lmul!(A::StridedArray,α::Number)
    A_dims = size(A)
    A = vec(A)
    n = length(A)
    # split up work
    nth = Threads.nthreads()
    chunks = get_chunks(n,nth)
    # do work 
    Threads.@threads for thn in 1:nth
        @simd for i in chunks[thn]:chunks[thn+1]
            @inbounds A[i] = α * A[i]
        end
    end
    reshape(A,A_dims)
end

function threaded_fill!(A::StridedArray,α::Number)
    A_dims = size(A)
    A = vec(A)
    n = length(A)
    # split up work
    nth = Threads.nthreads()
    chunks = get_chunks(n,nth)
    # do work 
    Threads.@threads for thn in 1:nth
        @simd for i in chunks[thn]:chunks[thn+1]
            @inbounds A[i] = α
        end
    end
    reshape(A,A_dims)
end

function threaded_mul!(C::StridedVecOrMat, A::AbstractSparseMatrixCSR, B::DenseInputVecOrMat, α::Number, β::Number)
    # john's implementation goes here
end




# y = α*transpose(A)*x + β*y
for (T, t) in ((Adjoint, adjoint), (Transpose, transpose))
    # treat csr as csc
    @eval function mul_col_major!(C::StridedVecOrMat, xA::$T{<:Any,<:AbstractSparseMatrixCSR}, B::DenseInputVecOrMat, α::Number, β::Number)
        A = xA.parent
        size(A, 2) == size(B, 1) || throw(DimensionMismatch())
        size(A, 1) == size(C, 1) || throw(DimensionMismatch())
        size(B, 2) == size(C, 2) || throw(DimensionMismatch())
        nzv = nonzeros(A)
        rv = rowvals(A)
        if β != 1
            β != 0 ? threaded_rmul!(C, β) : threaded_fill!(C, zero(eltype(C)))
        end
        
        for k in 1:size(C, 2) # loop over columns of input
            Threads.@threads for col in 1:size(A, 2)
                @inbounds αxj = B[col,k] * α
                @inbounds for j in nzrange(A, col)
                    val = $t(nzv[j])*αxj
                    row = rv[j]
                    out = Threads.Atomic{eltype(C)}(C[row, k])
                    Threads.atomic_add!(out, val)
                end
            end
        end
    end

    @eval function mul_row_major!(C::StridedVecOrMat, xA::$T{<:Any,<:AbstractSparseMatrixCSR}, B::DenseInputVecOrMat, α::Number, β::Number)
        A = xA.parent
        size(A, 2) == size(B, 1) || throw(DimensionMismatch())
        size(A, 1) == size(C, 1) || throw(DimensionMismatch())
        size(B, 2) == size(C, 2) || throw(DimensionMismatch())
        nzv = nonzeros(A)
        rv = rowvals(A)
        if β != 1
            β != 0 ? threaded_rmul!(C, β) : threaded_fill!(C, zero(eltype(C)))
        end
        
        Threads.@threads for col in 1:size(A, 2)
            @inbounds for j in nzrange(A, col)
                row = rv[j]
                Aα = $t(nzv[j]) * α
                for k in 1:size(C, 2) # loop over rows
                    val = B[col,k] * Aα
                    out = Threads.Atomic{eltype(C)}(C[row, k])
                    Threads.atomic_add!(out, val)
                end
            end
        end
    end

    @eval function mul!(C::StridedVecOrMat, xA::$T{<:Any,<:AbstractSparseMatrixCSR}, B::DenseInputVecOrMat, α::Number, β::Number)

        if stride(B,1) < stride(B,2) # implement different code based on strides
            # col major order
            mul_col_major!(C,xA,B,α,β)
        else
            # row major order
            mul_row_major!(C,xA,B,α,β)
        end
    end
end
