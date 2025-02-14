using BackwardsLinalg
using Test, Random
using Zygote

function gradient_check(f, args...; η = 1e-5)
    g = gradient(f, args...)
    dy_expect = η*sum(abs2.(g[1]))
    dy = f(args...)-f([gi === nothing ? arg : arg.-η.*gi for (arg, gi) in zip(args, g)]...)
    @show dy
    @show dy_expect
    isapprox(dy, dy_expect, rtol=1e-2, atol=1e-8)
end

@testset "lstsq" begin
    T = Float64
    Random.seed!(3)
    M, N = 10, 5
    A = randn(T, M, N)
    b = randn(T, M)
    op = randn(N, N)
    op += op'

    function tfunc(A, b)
        x = BackwardsLinalg.lstsq(A, b)
        return x'*op*x
    end
    tfuncA(A) = tfunc(A, b)
    tfuncb(b) = tfunc(A, b)
    @test gradient_check(tfuncA, A)
    @test gradient_check(tfuncb, b)
end
