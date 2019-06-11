using Test
@testset "BackwardsLinalg.jl" begin
    include("qr.jl")
    include("svd.jl")
    include("eigen.jl")
    #include("trg.jl")
end
