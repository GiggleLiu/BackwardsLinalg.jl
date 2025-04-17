using BackwardsLinalg
using Test, Random
using Zygote, FFTW

function gradient_check(f, args...; η = 1e-5)
    g = gradient(f, args...)
    dy_expect = η*sum(abs2.(g[1]))
    @show dy_expect
    dy = f(args...)-f([gi === nothing ? arg : arg.-η.*gi for (arg, gi) in zip(args, g)]...)
    @show dy
    isapprox(dy, dy_expect, rtol=1e-2, atol=1e-8)
end

@testset "FFT" begin
	Random.seed!(3)
    n = 8
    x = rand(ComplexF64, n)
	op = rand(ComplexF64, n, n)
	op = op + op'
    function tf(x)
		y = BackwardsLinalg.fft(op*x)
		return real(y'*op*y)
	end

    @test gradient_check(tf, x)
end



