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
@testset "lneq" begin
	T = ComplexF64
	Random.seed!(3)
	M = 10
	A = randn(T, M, M)
	b = randn(T, M)
	op = randn(T, M, M)
	op += op'

	function tfunc(A, b)
		x = BackwardsLinalg.lneq(A, b)
		return real(x' * op * x)
	end
	tfuncA(A) = tfunc(A, b)
	tfuncb(b) = tfunc(A, b)
	@test gradient_check(tfuncA, A)
	@test gradient_check(tfuncb, b)
end
