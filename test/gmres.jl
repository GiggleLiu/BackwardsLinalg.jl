using Zygote, LinearAlgebra, BackwardsLinalg
using Test, Random

function gradient_check(f, args...; η = 1e-5)
	g = gradient(f, args...)
	dy_expect = η * sum(abs2.(g[1]))
	@show dy_expect
	dy = f(args...) - f([gi === nothing ? arg : arg .- η .* gi for (arg, gi) in zip(args, g)]...)
	@show dy
	isapprox(dy, dy_expect, rtol = 1e-2)
end

# For dignal-domain matrix
@testset "gmres" begin
	Random.seed!(3)
	for T in [Float64, ComplexF64]
		n = 200
		A = rand(T, n, n) + n * LinearAlgebra.I
		b = rand(T, n)
		tf(A, b) = sum(abs2.(BackwardsLinalg.gmres(A, b)))
		tfA(A) = tf(A, b)
		tfb(b) = tf(A, b)

		@test gradient_check(tfA, A)
		@test gradient_check(tfb, b)
	end

end

# For less dignal-domain matrix
@testset "gmres falls" begin
	Random.seed!(3)
	for T in [Float64, ComplexF64]
		n = 100
		A = rand(T, n, n) + n / 32 * LinearAlgebra.I
		b = rand(T, n)
		tf(A, b) = sum(abs2.(BackwardsLinalg.gmres(A, b)))
		tfA(A) = tf(A, b)
		tfb(b) = tf(A, b)

		@test gradient_check(tfA, A)
		@test gradient_check(tfb, b)
	end

end
