using Test
using ParallelMergeCSR
using SparseArrays

# trigger merge_csr_mv!
@testset "Adjoint" begin
    
    # C = ABα + Cβ
end

# trigger merge_csr_mv!
@testset "Transpose" begin

    # C = ABα + Cβ
end

@testset "Matrix x Matrix" begin

    # C = ABα + Cβ
    ## A should be CSC matrix, B should be Dense
    ## α, β are `Number`s
    A = sprand(5, 5, 0.2)
    B = rand(5, 5)
    α = 1.1

    C = zeros(Float64, 5, 5)
    C_copy = deepcopy(C)
    β = 0.3

    SparseArrays.mul!(C, A, B, α, β)
    @test C ≈ Matrix(A) * B * α + C_copy * β

end

@testset "Matrix x Vector" begin

    A = sprand(5, 5, 0.3)
    B = rand(5)
    α = 2.5

    C = zeros(Float64, size(B))
    C_copy = deepcopy(C)
    β = 5.2

    SparseArrays.mul!(C, A, B, α, β)

    @test C ≈ Matrix(A) * B * α + C_copy * β
    

end