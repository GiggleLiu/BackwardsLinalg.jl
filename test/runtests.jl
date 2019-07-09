using Test

@testset "qr" begin
    include("qr.jl")
end

@testset "svd" begin
    include("svd.jl")
end

@testset "eigen" begin
    include("eigen.jl")
end
