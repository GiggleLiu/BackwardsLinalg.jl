using LinalgBackwards
using Test

@testset "qr" begin
    T = ComplexF64
    for (M, N) in [(4, 6), (4, 4), (6, 4)]
        A = randn(T, M, N)
        b2 = randn(T, N)
        b1 = randn(T, min(M, N))

        function tfunc(x)
            Q, R = qr(x)
            sum(Q*b1) + sum(R*b2) |> real
        end
        @test gradient_check(tfunc, A)

        a = [1 2; 3 4]
        @test copyltu!(a) == [1 3; 3 4]
    end
end

@testset "lq" begin
    for (M, N) in [(4, 6), (4, 4), (6, 4)]
        A = randn(M, N)
        b2 = randn(N)
        b1 = randn(min(M, N))

        function tfunc(x)
            Q, R = lq(x)
            sum(Q*b1) + sum(R*b2)
        end
        @test gradient_check(tfunc, A)

        a = [1 2; 3 4]
        @test copyltu!(a) == [1 3; 3 4]
    end
end
