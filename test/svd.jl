using LinalgBackwards
using Flux
using Flux.Tracker: @grad, data, track, TrackedTuple
using Test

@testset "svdbp" begin
    M, N = 4, 6
    K = min(M, N)
    A = randn(M, N)
    U, S, V = _svd(A)
    dU, dS, dV = randn(M, K), randn(K), randn(N, K)
    dA = svd_back(U, S, V, dU, dS, dV)

    for i in 1:length(A)
        δ = 0.01
        A[i] -= δ/2
        U1, S1, V1 = _svd(A)
        A[i] += δ
        U2, S2, V2 = _svd(A)
        A[i] -= δ/2
        δS = S2 .- S1
        δU = U2 .- U1
        δV = V2 .- V1
        @test isapprox(sum(dS .* δS) + sum(dU .* δU) + sum(dV .* δV), dA[i] * δ, atol=1e-5)
    end
end

@testset "svdflux" begin
    for (M, N) in [(4, 6), (4, 4), (6, 4)]
        K = min(M, N)
        A = randn(M, N)

        b = randn(K)
        function tfunc(x)
            U, S, V = _svd(x)
            sum(U.*b') + sum(V.*b') + sum(S'*b)
        end
        @test Tracker.gradcheck(tfunc, A)
    end
end
