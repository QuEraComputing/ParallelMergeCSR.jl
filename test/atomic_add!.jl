using Test
using ParallelMergeCSR
using SparseArrays

@testset "atomic_add!" begin
    # atomic add with real
    @testset "Real" begin
        a = rand(Float64, 20)
        a_copy = deepcopy(a)

        b = 1.25

        ParallelMergeCSR.atomic_add!(a, b)
        @test a == (a_copy .+ b)
    end

    # atomic add with complex
    @testset "Complex" begin
        a = rand(Complex{Float64}, 20)
        a_copy = deepcopy(a)

        b = 2.5 + 0.8im

        ParallelMergeCSR.atomic_add!(a, b)
        @test a == (a_copy .+ b)
    end
end