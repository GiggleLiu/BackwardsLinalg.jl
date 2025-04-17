using BackwardsLinalg
using Test, Random, LinearAlgebra
using Zygote

function gradient_check(f, args...; η = 1e-5)
	g = gradient(f, args...)
	dy_expect = η * sum(abs2.(g[1]))
	@show dy_expect
	dy = f(args...) - f([gi === nothing ? arg : arg .- η .* gi for (arg, gi) in zip(args, g)]...)
	@show dy
	isapprox(dy, dy_expect, rtol = 1e-2, atol = 1e-8)
end


@testset "symeigen for hermite" begin
    Random.seed!(6)
    T = ComplexF64
    n = 20
    A = randn(T, n, n)
    A = A+A'
    op = randn(T, n, n)
    op += op'
    function f(A)
        E, U = BackwardsLinalg.symeigen(A)
        E |> sum
    end
    function g(A)
        E, U = BackwardsLinalg.symeigen(A)
        v = U[:,end]
        (v'*op*v)[]|>real
    end
    @test gradient_check(f, A)
    @test gradient_check(g, A)
end