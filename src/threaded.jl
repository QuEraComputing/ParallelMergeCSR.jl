


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

#=
y = αAx + βy is expressed as C = αAx + βB 
C = ABα + Cβ
C can be a StridedVecOrMat
A will always be AbstractSparseMatrixCSR
B will be DenseInputVecOrMat

StridedVecOrMat is type in Base, Union of StridedVector and StridedMatrix w/ elems of type T
* StridedVector is a 1D StridedArray
* StridedMatrix is a 2D StridedArray 
* StridedArray can have elements stored at different positions in memory with fixed interval between elements,
  * all implement the strided array interface
* StrideArray is a subtype of AbtractArray, so getindex is defined as well

DenseInputVecOrMat
* Defined as: const DenseInputVecOrMat = Union{AdjOrTransDenseMatrix, DenseInputVector}
* DenseInputVector 
  * DenseInputVector is union of StridedVector and BitVector
* AdjOrTransDenseMatrix
  * Union of Adjoint, Transposed, or plain DenseMatrix
* DenseMatrix
  * Union of StridedMatrix, LowerTriangular, UnitLowerTriangular, UpperTriangular, UnitUpperTriangular, BitMatrix
=# 

#=
C = ABα + Cβ

A will always be an AbstractSparseMatrixCSR (Matrix)
B DenseInputVecOrMat (Vector or Matrix)
C StridedVecOrMat    (Vector or Matrix)
* If A and B are matrices, result will also be a matrix and C must also be Matrix
* if A is Matrix but B is vector, then C must be a vector as well


Split threaded_mul! into two functions via multiple dispatch: one handles Matrix-Vector, the other handles Matrix-Matrix
* threaded_mul!(C::StridedVector, A::AbstractSparseMatrixCSR, B::DenseInputVector, α::Number, β::Number)
* threaded_mul!(C::StridedMatrix, A::AbstractSparseMatrixCSR, B::AdjOrTransDenseMatrix)
=#
function threaded_mul!(C::StridedVecOrMat, A::AbstractSparseMatrixCSR, B::DenseInputVecOrMat, α::Number, β::Number)
    # john's implementation goes here
end

# Matrix × vector
function threaded_mul!(C::StridedVector, A::AbstractSparseMatrixCSR, B::DenseInputVector, α::Number, β::Number)
    # handle ABα
    ## calculate the shape of result ahead of time,
    ## if A is m x n, B must be n x 1, so result should be m x 1
    m, _ = size(A)
    mat_vec_result = zeros(m)
    merge_csr_mv!(A, B, mat_vec_result)
    mat_vec_result *= α

    # handle Cβ
    threaded_rmul!(C, β)
    # create AΒα + Cβ
    C += mat_vec_result
end

# Matrix × Matrix
function threaded_mul!(C::StridedMatrix, A::AbstractSparseMatrixCSR, B::AdjOrTransDenseMatrix, α::Number, β::Number)
    # handle ABα
    ## calculate the shape of result ahead of time
    ## if A is a x b and B is c x d, then result must be a x d
    a, _ = shape(A)
    _, d = shape(B)
    # fill()
    mat_mat_result = zeros(a,d)

    # repeatedly multiply A against each column in B, resulting 
    # in a row of mat_mat_result being produced
    for (row_idx, col) in enumerate(eachcol(B))
        merge_csr_mv!(A, col, mat_mat_result[row_idx])
    end

    # handle Cβ
    C *= β
    C += mat_mat_result
end

# handle the adjoint and transpose 
# C = adjoint(A)Bα + Cβ
# C = transpose(A)Bα + Cβ
# C = XAα + Cβ
# STAYS CSC bc adjoint(CSC) == CSR 
for (T, t) in ((Adjoint, adjoint), (Transpose, transpose))
    @eval function mul!(C::StridedVecOrMat, X::AdjOrTransDenseMatrix, xA::$T{<:Any,<:AbstractSparseMatrixCSR}, α::Number, β::Number)
        # obtains the original matrix underneath the "lazy wrapper"
        A = xA.parent
        # get dimensions of X
        mX, nX = size(X)
        # number of columns in X must be equal to number of colmuns in A
        nX == size(A, 2) || throw(DimensionMismatch())
        # number of rows in X must be equal to number of rows in C
        mX == size(C, 1) || throw(DimensionMismatch())
        # number of rows in A must be equal to number of columns in C
        size(A, 1) == size(C, 2) || throw(DimensionMismatch())
        rv = rowvals(A)
        nzv = nonzeros(A)
        # if beta isn't equal to 1...
            # if beta isn't equal to 0, then multiply C by beta but if it IS 0, 
            # then fill C with 
        if β != 1
            β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
        end
        @inbounds for col in 1:size(A, 2), k in nzrange(A, col)
            Aiα = $t(nzv[k]) * α
            rvk = rv[k]
            @simd for multivec_col in 1:mX
                C[multivec_col, rvk] += X[multivec_col, col] * Aiα
            end
        end
        C
    end
end
#= 
for (T, t) in ((Adjoint, adjoint), (Transpose, transpose))
    @eval function threaded_mul!(C::StridedVecOrMat, xA::$T{<:Any,<:AbstractSparseMatrixCSR}, B::DenseInputVecOrMat, α::Number, β::Number)
        A = xA.parent
        size(A, 2) == size(B, 1) || throw(DimensionMismatch())
        size(A, 1) == size(C, 1) || throw(DimensionMismatch())
        size(B, 2) == size(C, 2) || throw(DimensionMismatch())
        nzv = nonzeros(A)
        rv = rowvals(A)
        if β != 1
            β != 0 ? threaded_rmul!(C, β) : threaded_fill!(C, zero(eltype(C)))
        end
        if stride(C,1) < stride(C,2) # implement different code based on strides
            for k in 1:size(C, 2)
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
        else
            Threads.@threads for col in 1:size(A, 2)
                @inbounds for j in nzrange(A, col)
                    row = rv[j]
                    Aα = $t(nzv[j]) * α
                    for k in 1:size(C, 2)
                        val = B[col,k] * Aα
                        out = Threads.Atomic{eltype(C)}(C[row, k])
                        Threads.atomic_add!(out, val)
                    end
                end
            end
        end
    end
end
=# 