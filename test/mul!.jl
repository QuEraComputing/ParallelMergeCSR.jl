using Test
using ParallelMergeCSR
using SparseArrays

@testset "Sparse Matrix x Dense Vector" begin

    @testset "Adjoint Real Matrix" begin

        A = adjoint(sprand(10,10,0.5))

        B = rand(size(A, 2))
        α = -9.1

        C = rand(size(A, 2))
        C_copy = deepcopy(C)
        β = 3.9

        ParallelMergeCSR.mul!(C, A, B, α, β)
        SparseArrays.mul!(C_copy, A, B, α, β)
        @test C ≈ C_copy
    end

    @testset "Adjoint Complex Matrix" begin

        A = adjoint(sprand(Complex{Float64}, 10,10,0.5))

        B = rand(Complex{Float64}, size(A, 2))
        α = 30.5

        C = rand(Complex{Float64}, size(A, 2))
        C_copy = deepcopy(C)
        β = -6.2

        ParallelMergeCSR.mul!(C, A, B, α, β)
        SparseArrays.mul!(C_copy, A, B, α, β)
        @test C ≈ C_copy

    end

    @testset "Transpose Real Matrix" begin

        A = transpose(sprand(10,10,0.5))

        B = rand(size(A, 2))
        α = -9.1

        C = rand(size(A, 2))
        C_copy = deepcopy(C)
        β = 3.9

        ParallelMergeCSR.mul!(C, A, B, α, β)
        SparseArrays.mul!(C_copy, A, B, α, β)
        @test C ≈ C_copy
    end

    @testset "Transpose Complex Matrix" begin
        A = transpose(sprand(Complex{Float64},20,20,0.5))

        B = rand(Complex{Float64}, size(A, 2))
        α = -9.1

        C = rand(Complex{Float64}, size(A, 2))
        C_copy = deepcopy(C)
        β = 3.9

        ParallelMergeCSR.mul!(C, A, B, α, β)
        SparseArrays.mul!(C_copy, A, B, α, β)
        @test C ≈ C_copy
    end
end

# trigger merge_csr_mv! in this repo, does not default to mul! somewhere else
@testset "Sparse Matrix x Dense Random Matrix" begin

    @testset "Adjoint Real" begin

        A = adjoint(sprand(5, 5, 0.3))    
    
        B = rand(5, 5)
        α = 11.2
    
        C = rand(5, 5)
        C_copy = deepcopy(C)
        β = 3.9
    
        ParallelMergeCSR.mul!(C, A, B, α, β) 
        SparseArrays.mul!(C_copy, A, B, α, β)
        @test C ≈ C_copy
    
    end

    @testset "Adjoint Complex" begin
    
        # C = adjoint(A)Bα + Cβ
        A = adjoint(sprand(Complex{Float64}, 5, 5, 0.3))
        B = rand(5,5)
        α = 2.3
    
        C = rand(Complex{Float64}, 5, 5)
        C_copy = deepcopy(C)
        β = 1.2
    
        ParallelMergeCSR.mul!(C, A, B, α, β)
        SparseArrays.mul!(C_copy, A, B, α, β)
        @test C ≈ C_copy
    
    end

    @testset "Transpose Square Real" begin

        # C = transpose(A)Bα + Cβ
        A = transpose(sparse(rand(1:5, 4, 4)))
        B = rand(1:5, (4,4))
        α = 1.0
    
        C = zeros(4, 4)
        C_copy = deepcopy(C)
        β = 1.0
    
        ParallelMergeCSR.mul!(C, A, B, α, β)
        SparseArrays.mul!(C_copy, A, B, α, β)
        @test C ≈ C_copy

    
    end

    @testset "Transpose Square Complex" begin
        # C = transpose(A)Bα + Cβ
        A = transpose(sparse(rand(Complex{Float64}, 4, 4)))
        B = rand(4,4)
        α = 1.0
    
        C = zeros(eltype(A), 4, 4)
        C_copy = deepcopy(C)
        β = 6.11 + 9.2im
    
        ParallelMergeCSR.mul!(C, A, B, α, β)
        SparseArrays.mul!(C_copy, A, B, α, β)
        @test C ≈ C_copy
    end

    @testset "Transpose Rectangular Real" begin

        A = sparse(rand(4, 2)) |> transpose
        B = rand(4, 3)
    
        α = 9.1
    
        C = zeros(eltype(A), 2, 3)
        C_copy = deepcopy(C)
        β = 1.0
    
        ParallelMergeCSR.mul!(C, A, B, α, β)
        SparseArrays.mul!(C_copy, A, B, α, β)
        @test C ≈ C_copy
    
    end

    @testset "Transpose Rectangular Complex" begin

        A = sparse(rand(Complex{Float64}, 4, 2)) |> transpose
        B = rand(4, 3)
    
        α = 0.0 + 5.3im
    
        C = zeros(eltype(A), 2, 3)
        C_copy = deepcopy(C)
        β = 1.0
    
        ParallelMergeCSR.mul!(C, A, B, α, β)
        SparseArrays.mul!(C_copy, A, B, α, β)
        @test C ≈ C_copy
    
    end
end