using Test
using LinalgBackwards
using Random

@testset "svd grad U" begin
    function loss(A)
        M, N = size(A)
        U, S, V = svd(A)
        psi = U[:,1]
        Random.seed!(2)
        H = randn(ComplexF64, M, M)
        H+=H'
        real(psi'*H*psi)[]
    end

    for (M, N) in [(6, 3), (3, 6), (3,3)]
        K = min(M, N)
        a = randn(ComplexF64, M, N)
        @test gradient_check(loss, a)
    end
end


@testset "svd grad V" begin
    function loss_v(A)
        M, N = size(A)
        U, S, V = svd(A)
        Random.seed!(2)
        H = randn(ComplexF64, N, N)
        H+=H'
        psi = V[:,1]
        real(psi'*H*psi)[]
    end

    for (M, N) in [(6, 3), (3, 6), (3,3)]
        K = min(M, N)
        a = randn(ComplexF64, M,N)
        @test gradient_check(loss_v, a)
    end
end

@testset "svd grad S" begin
    function loss(A)
        U, S, V = svd(A)
        S |> sum
    end

    for (M, N) in [(6, 3), (3, 6), (3,3)]
        K = min(M, N)
        H1 = randn(ComplexF64, M, M)
        H1 += H1'
        a = randn(ComplexF64, M, N)
        @test gradient_check(loss, a)
    end
end

@testset "rsvd grad U" begin
    function loss(A)
        M, N = size(A)
        U, S, V = rsvd(A)
        psi = U[:,1]
        Random.seed!(2)
        H = randn(ComplexF64, M, M)
        H+=H'
        real(psi'*H*psi)[]
    end

    for (M, N) in [(6, 3), (3, 6), (3,3)]
        K = min(M, N)
        a = randn(ComplexF64, M, N)
        @test gradient_check(loss, a)
    end
end


@testset "rsvd grad V" begin
    function loss_v(A)
        M, N = size(A)
        U, S, V = rsvd(A)
        Random.seed!(2)
        H = randn(ComplexF64, N, N)
        H+=H'
        psi = V[:,1]
        real(psi'*H*psi)[]
    end

    for (M, N) in [(6, 3), (3, 6), (3,3)]
        K = min(M, N)
        a = randn(ComplexF64, M,N)
        @test gradient_check(loss_v, a)
    end
end

@testset "rsvd grad S" begin
    function loss(A)
        U, S, V = rsvd(A)
        S |> sum
    end

    for (M, N) in [(6, 3), (3, 6), (3,3)]
        K = min(M, N)
        H1 = randn(ComplexF64, M, M)
        H1 += H1'
        a = randn(ComplexF64, M, N)
        @test gradient_check(loss, a)
    end
end
