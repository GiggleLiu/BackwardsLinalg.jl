using BackwardsLinalg
using Test, Random

@testset "qr Q complex" begin
    T = ComplexF64
    Random.seed!(3)
    for (M, N) in [(2, 6), (4, 4), (6, 2)]
        A = randn(T, M, N)
        op = randn(M, M)
        op += op'
        op2 = randn(N, N)
        op2 += op2'

        function tfunc(x)
            Q, R = qr(x)
            v = Q[:,1]
            v2 = R[2,:]
            (v'*op*v + v2'*op2*v2)[] |> real
        end
        @test gradient_check(tfunc, A)
    end
    a = [1 2; 3 4]
    @test copyltu!(a) == [1 3; 3 4]
end

@testset "lq" begin
    Random.seed!(3)
    T = ComplexF64
    for (M, N) in [(2, 6), (4, 4), (6, 2)]
        @show M, N
        A = randn(T, M, N)

        op = randn(M, M)
        op += op'
        op2 = randn(N, N)
        op2 += op2'

        function tfunc(x)
            L, Q = lq(x)
            v = L[:,1]
            v2 = Q[2,:]
            (v'*op*v + v2'*op2*v2)[] |> real
        end
        @test gradient_check(tfunc, A)
    end
end
