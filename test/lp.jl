using BackwardsLinalg
using Test, Random
using Zygote

function gradient_check(f, args...; η = 1e-5)
	g = gradient(f, args...)
	g1 = g[1]
	dy_expect = (g1 === nothing ? 0.0 : η * sum(abs2.(g[1])))
	@show dy_expect
	dy = f(args...) - f([gi === nothing ? arg : arg .- η .* gi for (arg, gi) in zip(args, g)]...)
	@show dy
	isapprox(dy, dy_expect, rtol = 1e-2, atol = 1e-8)
end

@testset "standard lp" begin
	Random.seed!(3)
	M = 3
	N = 2
	η = 0.1
	c = [3.0, 2.0, 1.0] + (2 * rand(M) .- 1) * η
	A = [1.0 1.0 1.0; 2.0 1.0 0.0] + (2 * rand(N, M) .- 1) * η
	b = [4.0, 3.0] + (2 * rand(N) .- 1) * η

	function tfunc(c, A, b)
		x, a = BackwardsLinalg.lp(c, A, b)
		return sum(abs2.(x)) + a
	end

	tfuncc(c) = tfunc(c, A, b)
	tfuncA(A) = tfunc(c, A, b)
	tfuncb(b) = tfunc(c, A, b)

	@test gradient_check(tfuncc, c)
	@test gradient_check(tfuncA, A)
	@test gradient_check(tfuncb, b)
end


