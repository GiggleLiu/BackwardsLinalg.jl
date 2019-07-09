using BackwardsLinalg
using Random
using Test

@testset "symeigen real" begin
    A = randn(4,4)
    A = A+A'
    op = randn(4, 4)
    op += op'
    function f(A)
        E, U = symeigen(A)
        E |> sum
    end
    function g(A)
        E, U = symeigen(A)
        v = U[:,1]
        (v'*op*v)[]|>real
    end
    @test gradient_check(f, A)
    @test gradient_check(g, A)
end

@testset "symeigen complex" begin
    Random.seed!(6)
    A = randn(ComplexF64, 4,4)
    A = A+A'
    op = randn(ComplexF64, 4, 4)
    op += op'
    function f(A)
        E, U = symeigen(A)
        E |> sum
    end
    function g(A)
        E, U = symeigen(A)
        v = U[:,1]
        (v'*op*v)[]|>real
    end
    @test gradient_check(f, A)
    @test gradient_check(g, A)
end
