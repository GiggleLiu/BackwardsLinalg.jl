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


@testset "normeigen" begin
    Random.seed!(6)
    T = ComplexF64
    n = 20
    U = Matrix(LinearAlgebra.qr(randn(T, n, n)).Q)
    E = rand(T,n)
    op = randn(T, n, n)
    op += op'
    A = U * LinearAlgebra.Diagonal(E) * U'
    function f(A)
        E, U = BackwardsLinalg.normeigen(A)
        return sum(abs2.(E))
    end
    function g(A)
        E, U = BackwardsLinalg.normeigen(A)
        v = U[:,end]
        (v'*op*v)[]|>real
    end
    @test gradient_check(f, A)
    @test gradient_check(g, A)
end

@testset "normeigen for hermitian" begin
    Random.seed!(6)
    T = ComplexF64
    n = 20
    A = randn(T, n, n)
    A += A'
    op = randn(T, n, n)
    op += op'
    function f(A)
        E, U = BackwardsLinalg.normeigen(A)
        return sum(abs2.(E))
    end
    function g(A)
        E, U = BackwardsLinalg.normeigen(A)
        v = U[:,end]
        (v'*op*v)[]|>real
    end
    @test gradient_check(f, A)
    @test gradient_check(g, A)
end

