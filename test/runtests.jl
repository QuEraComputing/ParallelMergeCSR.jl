using Test

@testset "mv_mul" begin
    include("mv_mul.jl")
end

@testset "threaded" begin
    include("threaded.jl")
end