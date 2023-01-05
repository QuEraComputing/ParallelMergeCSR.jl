

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