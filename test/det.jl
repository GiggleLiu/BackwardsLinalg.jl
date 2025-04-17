using BackwardsLinalg
using Test, Random
using Zygote

function gradient_check(f, args...; η = 1e-5)
    g = gradient(f, args...)
    dy_expect = η*sum(abs2.(g[1]))
    @show dy_expect
    dy = f(args...)-f([gi === nothing ? arg : arg.-η.*gi for (arg, gi) in zip(args, g)]...)
    @show dy
    isapprox(dy, dy_expect, rtol=1e-2, atol=1e-8)
end


@testset "det" begin
    T = ComplexF64
    M = 4
    A = randn(T, M, M)
    function tfunc(A)
        a = BackwardsLinalg.det(A)
        return 2*abs2(a)-1
    end

    @test gradient_check(tfunc, A)
end

# When n>=6, Ā is to large to use finite difference. 
# We can just trust our AD rule is right 