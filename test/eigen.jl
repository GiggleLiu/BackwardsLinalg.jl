using BackwardsLinalg
using Random
using Test

function gradient_check(f, args...; η = 1e-5)
    g = gradient(f, args...)
    dy_expect = η*sum(abs2.(g[1]))
    dy = f(args...)-f([gi === nothing ? arg : arg.-η.*gi for (arg, gi) in zip(args, g)]...)
    @show dy
    @show dy_expect
    isapprox(dy, dy_expect, rtol=1e-2, atol=1e-8)
end

@testset "symeigen real" begin
    A = randn(4,4)
    A = A+A'
    op = randn(4, 4)
    op += op'
    function f(A)
        E, U = BackwardsLinalg.symeigen(A)
        E |> sum
    end
    function g(A)
        E, U = BackwardsLinalg.symeigen(A)
        v = U[:,1]
        (v'*op*v)[]|>real
    end
    @test gradient_check(f, A)
    @test gradient_check(g, A)
end

@testset "symeigen complex" begin
    Random.seed!(6)
    A = randn(ComplexF64, 4,4)
    A = A+A'
    op = randn(ComplexF64, 4, 4)
    op += op'
    function f(A)
        E, U = BackwardsLinalg.symeigen(A)
        E |> sum
    end
    function g(A)
        E, U = BackwardsLinalg.symeigen(A)
        v = U[:,1]
        (v'*op*v)[]|>real
    end
    @test gradient_check(f, A)
    @test gradient_check(g, A)
end
