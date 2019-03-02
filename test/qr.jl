include("../src/qr.jl")
using Test

@testset "_qr" begin
    M, N = 4, 6
    A = randn(M, N)
    b2 = randn(N)
    b1 = randn(M)

    @test Tracker.gradcheck(x->sum(_qr(x)[2]*b2) + sum(_qr(x)[1]*b1), A)

    a = [1 2; 3 4]
    @test copyltu!(a) == [1 3; 3 4]
end
