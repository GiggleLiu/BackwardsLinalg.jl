using LinalgBackwards
using Flux
using Flux.Tracker: @grad, data, track, TrackedTuple
using Test

@testset "diageigbp" begin
    N = 4
    A0 = randn(N, N)
    A = (A0 + A0') / 2
    λ, U = _diageig(A)
    dλ, dU = randn(N), randn(N, N)
    dA = diageig_back(λ, U, dλ, dU)

    for i in 1:length(A)
        δ = 0.001
        A0[i] -= δ
        A = (A0 + A0') / 2
        λ1, U1 = _diageig(A)

        A0[i] += 2 * δ
        A = (A0 + A0') / 2
        λ2, U2 = _diageig(A)

        A0[i] -= δ
        A = (A0 + A0') / 2
        δλ = λ2 .- λ1
        δU = U2 .- U1
        @test isapprox(sum(dλ .* δλ) + sum(dU .* δU), dA[i] * δ, atol=1e-5)
    
    end
end