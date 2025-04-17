using BackwardsLinalg, NFFT
using Test, Random, LinearAlgebra
using Zygote

function gradient_check(f, args...; η = 1e-8)
	g = gradient(f, args...)
	dy_expect = η * sum(abs2.(g[1]))
	@show dy_expect
	dy = f(args...) - f([gi === nothing ? arg : arg .- η .* gi for (arg, gi) in zip(args, g)]...)
	@show dy
	isapprox(dy, dy_expect, rtol = 1e-1)
end

@testset "unfft" begin
	Random.seed!(3)
	N = 32
	k = rand(N) .- 0.5
	f = rand(ComplexF64, N)
	tf(x) = sum(abs2.(BackwardsLinalg.unfft(k, x)))
	@test gradient_check(tf, f)
end


@testset "iunfft_t2" begin
	Random.seed!(3)
	N = 10
	k = rand(N) .- 0.5
	f = rand(ComplexF64, N)
	A = BackwardsLinalg.A_construct_t2(k)
	fhat = A * f
	tf(x) = sum(abs2.(BackwardsLinalg.inufft_t2(k, x)))
	@test gradient_check(tf, fhat)
end






