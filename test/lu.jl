using BackwardsLinalg
using Test, Random
using Zygote
using LinearAlgebra

function gradient_check(f, args...; η = 1e-5)
    g = gradient(f, args...)
    dy_expect = η*sum(abs2.(g[1]))
    @show dy_expect
    dy = f(args...)-f([gi === nothing ? arg : arg.-η.*gi for (arg, gi) in zip(args, g)]...)
    @show dy
    isapprox(dy, dy_expect, rtol=1e-2, atol=1e-8)
end

@testset "lu" begin
    Random.seed!(3)
    T = ComplexF64
    M =5
    A = rand(T,M,M)
    function tfunc(A)
        L,U,_ = BackwardsLinalg.lu(A)
        return sum(abs2.(L[:,1]'*U[:,end]))
    end

    @test gradient_check(tfunc,A)

end




