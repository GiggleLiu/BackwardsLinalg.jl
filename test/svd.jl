using LinalgBackwards
using Flux
using Flux.Tracker: @grad, data, track, TrackedTuple
using Test

@testset "svdbp" begin
    M, N = 4, 6
    T = ComplexF64
    K = min(M, N)
    A = randn(T, M, N)
    U, S, V = svd(A)
    dU, dS, dV = randn(T, M, K)|>real, zeros(real(T), K), zeros(T, N, K)
    dA = svd_back(U, S, V, dU, dS, dV)
    H = randn(ComplexF64, 4,4); H=H+H'
    function loss(A)
        U, S, V = svd(A)
        psi = U[:,1]
        psi'*H*psi
    end
    function gradient(A, y)

    for i in 1:length(A)
        δ = 0.01 +0.01im
        A[i] -= δ/2
        U1, S1, V1 = svd(A)
        A[i] += δ
        U2, S2, V2 = svd(A)
        A[i] -= δ/2
        δS = S2 .- S1
        δU = U2 .- U1
        δV = V2 .- V1
        @test isapprox(sum(dS .* δS) + sum(dU .* δU) + sum(dV .* δV) |> real, dA[i] * δ |> real, atol=1e-4)
        #@test isapprox(sum(dS .* δS) + sum(dU .* δU) + sum(dV .* δV), dA[i] * δ, atol=1e-3)
    end
end

import Tracker: gradcheck, ngradient

function gradcheck(f, xs...; rtol=1e-5, atol=1e-5, δ=1e-5)
    ng = ngradient(f, xs...; δ=δ)
    ag = data.(Tracker.gradient(f, xs...))
    @show ng
    @show ag
    all(isapprox.(ng, ag, rtol = rtol, atol = atol))
end

function ngradient(f, xs::AbstractArray...; δ=1e-5)
  grads = zero.(xs)
  for (x, Δ) in zip(xs, grads), i in 1:length(x)
    δ = sqrt(eps())
    tmp = x[i]
    x[i] = tmp - δ/2
    y1 = f(xs...)
    x[i] = tmp + δ/2
    y2 = f(xs...)
    x[i] = tmp
    Δ[i] = (y2-y1)/δ
  end
  return grads
end

@testset "svdflux" begin
    for (M, N) in [(4, 6), (4, 4), (6, 4)]
        K = min(M, N)
        T = ComplexF64
        A = randn(T, M, N)

        b = randn(T, K)
        op = randn(K, K)
        function tfunc(x)
            U, S, V = svd(x)
            psi = U[:,1]
            psi |> real
        end
        #@show tfunc(A)
        @test gradcheck(tfunc, A, rtol=1e-3, atol=1e-3, δ=1e-4)
    end
end

@testset "svdbp-complex-U" begin
    M, N = 4, 6
    T = ComplexF64
    K = min(M, N)
    A = randn(T, M, N)
    H = randn(ComplexF64, M, M)
    H = randn(M, M)
    H=H+H'
    function loss(A)
        U, S, V = svd(A)
        psi = U[:,1]
        psi'*H*psi
    end
    function gradient(A)
        U, S, V = svd(A)
        dU = zero(U)
        dS = zero(S)
        dV = zero(V)
        dU[:,1] = U[:,1]'*H
        dA = svd_back(U, S, V, dU, dS, dV)
        dA
    end

    ag = gradient(A)
    for i in 1:length(A)
        δ = 0.001*randn(T)
        A[i] -= δ/2
        f1 = loss(A)
        A[i] += δ
        f2 = loss(A)
        A[i] -= δ/2
        ng = (f2-f1)
        @test isapprox(ng, ag[i]*δ+conj(ag[i])*conj(δ), atol=1e-7, rtol=1e-7)
        @show ng, ag[i]*δ+conj(ag[i])*conj(δ)
    end
end

@testset "svdbp-complex-U" begin
    M, N = 6, 4
    T = ComplexF64
    K = min(M, N)
    A = randn(T, M, N)
    H = randn(T, M, M)
    H=H+H'
    function loss(A)
        U, S, V = svd(A)
        psi = U[:,1]
        psi'*H*psi
    end
    function gradient(A)
        U, S, V = svd(A)
        dU = zero(U)
        dS = zero(S)
        dV = zero(V)
        dU[:,1] = U[:,1]'*H
        dA = svd_back(U, S, V, dU, dS, dV)
        dA
    end

    ag = gradient(A)
    for i in 1:length(A)
        δ = 0.001*randn(T)
        A[i] -= δ/2
        f1 = loss(A)
        A[i] += δ
        f2 = loss(A)
        A[i] -= δ/2
        ng = (f2-f1)
        @test isapprox(ng, ag[i]*δ+conj(ag[i])*conj(δ), atol=1e-7, rtol=1e-7)
        @show ng, ag[i]*δ+conj(ag[i])*conj(δ)
    end
end


@testset "svdbp-complex-V" begin
    M, N = 4, 6
    T = ComplexF64
    K = min(M, N)
    A = randn(T, M, N)
    H = randn(T, N,N)
    H=H+H'
    function loss(A)
        U, S, V = svd(A)
        psi = V[:,1]
        psi'*H*psi
    end
    function gradient(A)
        U, S, V = svd(A)
        dU = zero(U)
        dS = zero(S)
        dV = zero(V)
        dV[:,1] = V[:,1]'*H
        dA = svd_back(U, S, V, dU, dS, dV)
        dA
    end

    ag = gradient(A)
    for i in 1:length(A)
        δ = 0.001*randn(T)
        A[i] -= δ/2
        f1 = loss(A)
        A[i] += δ
        f2 = loss(A)
        A[i] -= δ/2
        ng = (f2-f1)
        @test isapprox(ng, ag[i]*δ+conj(ag[i])*conj(δ), atol=1e-7, rtol=1e-7)
        #@show ng, ag[i]*δ+conj(ag[i])*conj(δ)
    end
end

@testset "svdbp-complex-S" begin
    M, N = 4, 6
    T = ComplexF64
    K = min(M, N)
    A = randn(T, M, N)
    function loss(A)
        U, S, V = svd(A)
        S |> sum
    end
    function gradient(A)
        U, S, V = svd(A)
        dU = zero(U)
        dS = similar(S)
        dS .= 1
        dV = zero(V)
        dA = svd_back(U, S, V, dU, dS, dV)
        dA
    end

    ag = gradient(A)
    for i in 1:length(A)
        δ = 0.001*randn(T)
        A[i] -= δ/2
        f1 = loss(A)
        A[i] += δ
        f2 = loss(A)
        A[i] -= δ/2
        ng = (f2-f1)
        @test isapprox(ng, ag[i]*δ+conj(ag[i])*conj(δ), atol=1e-7, rtol=1e-7)
        #@show ng, ag[i]*δ+conj(ag[i])*conj(δ)
    end
end
