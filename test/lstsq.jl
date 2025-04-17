using BackwardsLinalg
using Test, Random
using Zygote


function gradient_check(f, args...; η = 1e-5)
	g = gradient(f, args...)
	dy_expect = η * sum(abs2.(g[1]))
	@show dy_expect
	dy = f(args...) - f([gi === nothing ? arg : arg .- η .* gi for (arg, gi) in zip(args, g)]...)
	@show dy
	isapprox(dy, dy_expect, rtol = 1e-2, atol = 1e-8)
end

# test for real and complex
@testset "arg_lstsq" begin
	T = ComplexF64
	Random.seed!(3)
	M, N = 10, 5
	A = randn(T, M, N)
	b = randn(T, M)
	op = randn(T, N, N)
	op += op'

	function tfunc(A, b)
		x = BackwardsLinalg.arg_lstsq(A, b)
		return real(x' * op * x)
	end
	tfuncA(A) = tfunc(A, b)
	tfuncb(b) = tfunc(A, b)
	@test gradient_check(tfuncA, A)
	@test gradient_check(tfuncb, b)
end

@testset "lstsq" begin
	T = ComplexF64
	Random.seed!(3)
	M, N = 10, 5
	A = randn(T, M, N)
	b = randn(T, M)
	op = randn(T, N, N)
	op += op'

	function tfunc(A, b)
		a = BackwardsLinalg.lstsq(A, b)
		return 2*(a-1.0)^2-1.0
	end

	@show BackwardsLinalg.lstsq(A,b)
	tfuncA(A) = tfunc(A, b)
	tfuncb(b) = tfunc(A, b)
	@test gradient_check(tfuncA, A)
	@test gradient_check(tfuncb, b)
end