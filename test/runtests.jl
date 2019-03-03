using Test
@testset "LinalgBackwards.jl" begin
    include("svd.jl")
    include("qr.jl")
    #include("trg.jl")
end
