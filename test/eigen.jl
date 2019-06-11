using BackwardsLinalg
using Random
using Test

@testset "eigen real" begin
    A = randn(4,4)
    A = A+A'
    op = randn(4, 4)
    op += op'
    function f(A)
        E, U = eigen(A)
        E |> sum
    end
    function g(A)
        E, U = eigen(A)
        v = U[:,1]
        (v'*op*v)[]|>real
    end
    @test gradient_check(f, A)
    @test gradient_check(g, A)
end

@testset "eigen complex" begin
    Random.seed!(6)
    A = randn(ComplexF64, 4,4)
    A = A+A'
    op = randn(ComplexF64, 4, 4)
    op += op'
    function f(A)
        E, U = eigen(A)
        E |> sum
    end
    function g(A)
        E, U = eigen(A)
        v = U[:,1]
        (v'*op*v)[]|>real
    end
    @test gradient_check(f, A)
    @test gradient_check(g, A)
end
