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

@testset "cls" begin
    T = ComplexF64

    function tfunc(A)
        L = BackwardsLinalg.cls(A)
        return sum(abs2.(L[:,1]))
    end

    M = 10
    A = randn(T, M, M)
    A = A' * A

    @test gradient_check(tfunc,A)

end

