using BackwardsLinalg
using Test, Random
using Zygote


function gradient_check(f, args...; η = 1e-5)
    g = gradient(f, args...)
    dy_expect = η*sum(abs2.(g[1]))
    dy = f(args...)-f([gi === nothing ? arg : arg.-η.*gi for (arg, gi) in zip(args, g)]...)
    isapprox(dy, dy_expect, rtol=1e-2, atol=1e-8)
end

@testset "mxmul" begin
    T = ComplexF64
    Random.seed!(3)
    times = 6
    M = 10 * times
    N = 5 * times
    K = 8 * times
    A = rand(T, M, N)
    B = rand(T, N, K)

    function tfunc(A,B)
        C = BackwardsLinalg.mxmul(A,B)
        return sum(abs2.(C[1,:]))
    end

    tfuncA(A) = tfunc(A, B)
    tfuncB(B) = tfunc(A, B)
    @test gradient_check(tfuncA, A)
    @test gradient_check(tfuncB, B)
end