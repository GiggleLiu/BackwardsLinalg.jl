using BackwardsLinalg
using Test, Random
using Zygote, LinearAlgebra


function gradient_check(f, args...; η = 1e-5)
	g = gradient(f, args...)
	dy_expect = η * sum(abs2.(g[1]))
	@show dy_expect
	dy = f(args...) - f([gi === nothing ? arg : arg .- η .* gi for (arg, gi) in zip(args, g)]...)
	@show dy
	isapprox(dy, dy_expect, rtol = 1e-2, atol = 1e-8)
end

@testset "pf" begin
    Random.seed!(3)
    T = Float64
    n = 10
    A = rand(T,n,n)
    A -= A'
    tf(A) = BackwardsLinalg.pf(A)^2 - 1.0

    @test gradient_check(tf,A)
end

