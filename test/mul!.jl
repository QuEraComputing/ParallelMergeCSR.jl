using Test
using ParallelMergeCSR
using SparseArrays

# trigger merge_csr_mv! in this repo, does not default to mul! somewhere else
@testset "Adjoint Real" begin

    A = adjoint(sprand(5, 5, 0.3))    

    B = rand(5, 5)
    α = 11.2

    C = rand(5, 5)
    C_copy = deepcopy(C)
    β = 3.9

    @test ParallelMergeCSR.mul!(C, A, B, α, β) ≈ SparseArrays.mul!(C_copy, A, B, α, β)

end

@testset "Adjoint Complex" begin
    
    # C = adjoint(A)Bα + Cβ
    A = adjoint(sprand(Complex{Float64}, 5, 5, 0.3))
    B = rand(5,5)
    α = 2.3

    C = rand(Complex{Float64}, 5, 5)
    C_copy = deepcopy(C)
    β = 1.2

    @test ParallelMergeCSR.mul!(C, A, B, α, β) ≈ SparseArrays.mul!(C_copy, A, B, α, β)

end
# trigger merge_csr_mv!

@testset "Transpose Square Real" begin

    # C = transpose(A)Bα + Cβ
    A = transpose(sparse(rand(1:5, 4, 4)))
    B = rand(1:5, (4,4))
    α = 1.0

    C = zeros(4, 4)
    C_copy = deepcopy(C)
    β = 1.0

    # right hand side is correct, left hand side is problematic
    @test ParallelMergeCSR.mul!(C, A, B, α, β) ≈ SparseArrays.mul!(C_copy, A, B, α, β)

end


@testset "Transpose Square Complex" begin
    # C = transpose(A)Bα + Cβ
    A = transpose(sparse(rand(Complex{Float64}, 4, 4)))
    B = rand(4,4)
    α = 1.0

    C = zeros(eltype(A), 4, 4)
    C_copy = deepcopy(C)
    β = 6.11 + 9.2im


    @test ParallelMergeCSR.mul!(C, A, B, α, β) ≈ SparseArrays.mul!(C_copy, A, B, α, β)
end

@testset "Transpose Rectangular Real" begin

    A = sparse(rand(4, 2)) |> transpose
    B = rand(4, 3)

    α = 9.1

    C = zeros(eltype(A), 2, 3)
    C_copy = deepcopy(C)
    β = 1.0

    @test ParallelMergeCSR.mul!(C, A, B, α, β) ≈ SparseArrays.mul!(C_copy, A, B, α, β)

end

@testset "Transpose Rectangular Complex" begin

    A = sparse(rand(Complex{Float64}, 4, 2)) |> transpose
    B = rand(4, 3)

    α = 0.0 + 5.3im

    C = zeros(eltype(A), 2, 3)
    C_copy = deepcopy(C)
    β = 1.0

    @test ParallelMergeCSR.mul!(C, A, B, α, β) ≈ SparseArrays.mul!(C_copy, A, B, α, β)

end

@testset "Matrix x Matrix (Real)" begin

    # A should be sparse, B should be dense
    A = sprand(5, 5, 0.2)
    B = rand(5, 5)
    α = 1.1

    C = zeros(5, 5)
    C_copy = deepcopy(C)
    β = 0.3

    @test ParallelMergeCSR.mul!(C, A, B, α, β) ≈ SparseArrays.mul!(C_copy, A, B, α, β)

end

@testset "Matrix x Matrix (Complex)" begin

    # A should be sparse, B should be dense
    A = sprand(5, 5, 0.2)
    B = rand(Complex{Float64}, 5, 5)
    α = 1.1

    C = zeros(eltype(B), 5, 5)
    C_copy = deepcopy(C)
    β = 0.3

    @test ParallelMergeCSR.mul!(C, A, B, α, β) ≈ SparseArrays.mul!(C_copy, A, B, α, β)

end

@testset "Matrix x Vector (Real)" begin

    A = sprand(5, 5, 0.3)
    B = rand(5)
    α = 2.5

    C = zeros(Float64, size(B))
    C_copy = deepcopy(C)
    β = 5.2

    @test ParallelMergeCSR.mul!(C, A, B, α, β) ≈ SparseArrays.mul!(C_copy, A, B, α, β)
end

@testset "Matrix x Vector (Real)" begin

    A = sprand(5, 5, 0.3)
    B = rand(5)
    α = 2.5

    C = zeros(Float64, size(B))
    C_copy = deepcopy(C)
    β = 5.2

    @test ParallelMergeCSR.mul!(C, A, B, α, β) ≈ SparseArrays.mul!(C_copy, A, B, α, β)

end

@testset "Matrix x Vector (Complex)" begin

    A = sprand(5, 5, 0.3)
    B = rand(Complex{Float64}, 5)
    α = 2.5

    C = zeros(eltype(B), size(B))
    C_copy = deepcopy(C)
    β = 2.1+0.1im

    @test ParallelMergeCSR.mul!(C, A, B, α, β) ≈ SparseArrays.mul!(C_copy, A, B, α, β)

end