using LinalgBackwards
using Test

@testset "_qr" begin
    for (M, N) in [(4, 6), (4, 4), (6, 4)]
        A = randn(M, N)
        b2 = randn(N)
        b1 = randn(min(M, N))

        function tfunc(x)
            Q, R = _qr(x)
            sum(Q*b1) + sum(R*b2)
        end
        @test Tracker.gradcheck(tfunc, A)

        a = [1 2; 3 4]
        @test copyltu!(a) == [1 3; 3 4]
    end
end
