using BackwardsLinalg
using Test, Random
using Zygote

function gradient_check(f, args...; η = 1e-5)
    g = gradient(f, args...)
    dy_expect = η*sum(abs2.(g[1]))
    dy = f(args...)-f([gi === nothing ? arg : arg.-η.*gi for (arg, gi) in zip(args, g)]...)
    isapprox(dy, dy_expect, rtol=1e-2, atol=1e-8)
end

@testset "scha_norm" begin
    T = ComplexF64
    Random.seed!(3)
    M = 10
    N = 5
    A = randn(T, M, N)
    function tfunc(A ,p)
        a = BackwardsLinalg.scha_norm(A ,p)
        return 2 * a -1
    end

    p = 2.0
    @test gradient_check(tfunc, A, p)

    p = Inf
    @test gradient_check(tfunc, A, p)
end

