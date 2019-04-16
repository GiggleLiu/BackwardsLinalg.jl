using Test
@testset "LinalgBackwards.jl" begin
    include("qr.jl")
    include("svd_gradient.jl")
    #include("trg.jl")
end
