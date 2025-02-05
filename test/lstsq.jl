using BackwardsLinalg
using Test, Random

@testset "lstsq" begin
    T = Float64
    Random.seed!(3)
    M, N = 10, 5
    A = randn(T, M, N)
    b = randn(T, M)
    op = randn(N, N)
    op += op'

    function tfunc(A, b)
        x = lstsq(A, b)
        return x'*op*x
    end
    tfuncA(A) = tfunc(A, b)
    tfuncb(b) = tfunc(A, b)
    @test gradient_check(tfuncA, A)
    @test gradient_check(tfuncb, b)
end
