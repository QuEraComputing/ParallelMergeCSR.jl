using Test

@testset "merge_csr_mv!" begin
    include("merge_csr_mv!.jl")
end

@testset "mul!" begin
    include("mul!.jl")
end