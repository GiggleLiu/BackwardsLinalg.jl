using LinearAlgebra

function rsvd(A::Array{T}, k::Int, oversample=10, power=10, ortho=false) where T
    m, n = size(A)
    p = min(n,oversample*k)
    Y = A * randn(T, n,p)

    for i = 1:power
        Y = A * (ortho ? qr(A' * Y).Q : A' * Y)
    end
    Q, R = qr(Y)
    B = Q' * A

    U, s ,V = svd(B)
    U = Q * U
    k = min(k,size(U, 2))
    U = U[:,1:k]
    V = V[:,1:k]
    s = s[1:k]
    return U,s,V
end

using Test
@testset "svd" begin
    for shape in [(100, 30), (30, 30), (30, 100)]
        A = randn(shape...)
        U, S, V = rsvd(A, 30)
        @test U*Diagonal(S)*V' ≈ A
    end

    A = randn(100, 30) * randn(30, 70)
    U, S, V = rsvd(A, 30)
    @test U*Diagonal(S)*V' ≈ A
end
