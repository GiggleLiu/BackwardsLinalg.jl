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

@testset "norm_anlfunc" begin
	Random.seed!(3)
	T = ComplexF64
	M = 10
	S = randn(T, M)
	Q,R = LinearAlgebra.qr( randn(T, M, M) )
    Q = Matrix(Q)
	A = Q * diagm(S) * Q'
	f1(x) = x^2 - 4 * x + 2.0
	df1(x) = 2 * x - 4.0
	f2(x) = exp(2 * x - 1) + 2.0
	df2(x) = 2 * exp(2 * x - 1)
	tfunc1(A) = sum(abs2.(BackwardsLinalg.norm_anlfunc(f1, df1, A)))
	tfunc2(A) = sum(abs2.(BackwardsLinalg.norm_anlfunc(f2, df2, A)))

	@test gradient_check(tfunc1, A)
	@test gradient_check(tfunc2, A)
end
