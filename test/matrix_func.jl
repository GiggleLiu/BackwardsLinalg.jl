using Random,Test,LinearAlgebra,Zygote
using BackwardsLinalg

function gradient_check(f, args...; η = 1e-5)
    g = gradient(f, args...)
    dy_expect = η*sum(abs2.(g[1]))
    dy = f(args...)-f([gi === nothing ? arg : arg.-η.*gi for (arg, gi) in zip(args, g)]...)
    isapprox(dy, dy_expect, rtol=1e-2, atol=1e-8)
end

@testset "matrix_func" for T in [Float32,Float64,ComplexF32,ComplexF64]
    n = 10
    Random.seed!(6)
    A = rand(T,n,n)
    @testset for f in [x->exp(x),x->sin(x),x->x^2-x+I]
        @test isapprox(f(A),BackwardsLinalg.matrix_func(f,A);rtol = 1e-2)
    end
end

@testset "gradient of matrix_func" for T in [Float32,Float64,ComplexF32,ComplexF64]
    T = Float64
    n = 10
    Random.seed!(6)
    A = rand(T,n,n)
    @testset for f in [x->exp(x),x->sin(x),x->x^2-x+I]
        f = x->exp(x)
        testf(A) = tr(BackwardsLinalg.matrix_func(f,A))
        @test gradient_check(testf,A)
    end
end


