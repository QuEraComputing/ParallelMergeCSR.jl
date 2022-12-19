using Test
using ParallelMergeCSR
using SparseArrays

# works with AbstractSparseMatrixCSC or just plain AbstractMatrix
m_to_csr(M::AbstractMatrix) =  M |> SparseArrays.sparse |> ParallelMergeCSR.SparseMatrixCSR

@testset "Extreme Cases" begin

    @testset "Singleton" begin
        full_m = reshape([1], 1, 1)
        csr_m = m_to_csr(full_m)

        x = rand(1)

        y = zeros(csr_m.m)

        ParallelMergeCSR.merge_csr_mv!(csr_m, x, y)

        @test full_m * x == y
    end

    @testset "Single row" begin

        csc_m = SparseArrays.sprand(10, 1, 0.3)
        csr_m = m_to_csr(csc_m)

        x = rand(1:10, 1)

        y = zeros(csr_m.m)

        ParallelMergeCSR.merge_csr_mv!(csr_m, x, y)

        @test csc_m * x ≈ y
    end

    @testset "Single column" begin

        csc_m = SparseArrays.sprand(10, 1, 0.3)
        csr_m = m_to_csr(csc_m)

        x = rand(1:10, 1)

        y = zeros(csr_m.m)

        ParallelMergeCSR.merge_csr_mv!(csr_m, x, y)

        @test csc_m * x ≈ y
    end
end

@testset "Square" begin
    # 10 x 10 with 30% chance of entry being made
    full_m = SparseArrays.sprand(10,10,0.3)
    csr_m = m_to_csr(full_m)

    x = rand(1:10, 10, 1)

    y = zeros(csr_m.m)

    ParallelMergeCSR.merge_csr_mv!(csr_m, x, y)

    @test full_m * x ≈ y

end

@testset "4x6" begin
    # create matrix
    full_m = [10 20 0 0 0 0;
              0 30 0 40 0 0;
              0 0 50 60 70 0;
              0 0 0 0 0 80]

    # get into CSR form
    csr_m = m_to_csr(full_m)

    # create vector
    x = [5,2,3,1,8,2]

    # create empty solution
    y = zeros(Int64, csr_m.m)
    
    # multiply
    ParallelMergeCSR.merge_csr_mv!(csr_m, x, y)

    @test full_m * x == y
end