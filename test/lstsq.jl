using BackwardsLinalg
using Test, Random

@testset "lstsq" begin
    T = Float64
    Random.seed!(3)
    M, N = 20, 5
    A = randn(T, M, N)
    b = randn(T, M)
    op = randn(N, N)
    op += op'

    function tfunc(A, b)
        x = lstsq(A, b)
        return x'*op*x
    end
    @test gradient_check(tfunc, A, b)
end