using Test
@testset "LinalgBackwards.jl" begin
    include("qr.jl")
    include("svd.jl")
    include("eigen.jl")
    #include("trg.jl")
end
