# API for reference
using CuArrays
using CuArrays.CUSOLVER
CuArrays.allowscalar(false)  # important to prevent element-wise operation to GPU Arrays directly.

a = randn(10, 10) |> CuArray

using CUDAnative
CUDAnative.device!(0)

using BackwardsLinalg
import BackwardsLinalg: svd_back, qr_back, symeigen_back
using LinearAlgebra

import LinearAlgebra: eigen
function BackwardsLinalg.symeigen(a::CuArray{<:Complex})
    CUSOLVER.heevd!('V','U',copy(a))
end

function BackwardsLinalg.symeigen(a::CuArray{<:Real})
    CUSOLVER.seevd!('V','U',copy(a))
end

function LinearAlgebra.qr(a::CuArray)
    M, N = size(a)
    A, tau = CUSOLVER.geqrf!(copy(a))
    if M > N
        R = triu(A[1:N,:])
    else
        R = triu(A)
    end
    Q = CUSOLVER.orgqr!(A, tau)
    return Q, R
end

@testset "cuda qr" begin
    for T in [ComplexF32, Float64]
        for size in [(2,4), (4,2), (4,4)]
            @show size
            a = CuArray(randn(T, size...))
            Q, R = qr(a)
            @test Matrix(Q'*Q) ≈ I
            @test Q*R ≈ a
            @test a ≈ a
        end
    end
end

using Test

using CUDAnative
import CUDAnative: pow, abs, angle
for (RT, CT) in [(:Float64, :ComplexF64), (:Float32, :ComplexF32)]
    @eval CUDAnative.angle(z::$CT) = CUDAnative.atan2(CUDAnative.imag(z), CUDAnative.real(z))
    @eval function CUDAnative.abs(z::$CT)
        i = CUDAnative.imag(z)
        r = CUDAnative.real(z)
        CUDAnative.sqrt(i*i+r*r)
    end

    @eval cp2c(d::$RT, a::$RT) = CUDAnative.ComplexF64(d*CUDAnative.cos(a), d*CUDAnative.sin(a))
    for NT in [RT, :Int32, :Int64]
        @eval CUDAnative.pow(z::$CT, n::$NT) = CUDAnative.ComplexF64((CUDAnative.pow(CUDAnative.abs(z), n)*CUDAnative.cos(n*CUDAnative.angle(z))), (CUDAnative.pow(CUDAnative.abs(z), n)*CUDAnative.sin(n*CUDAnative.angle(z))))
    end
end

function BackwardsLinalg.svd_back(U::AbstractArray{T}, S, V, dU, dS, dV; η::Real=1e-40) where T
    all(x -> x isa Nothing, (dU, dS, dV)) && return nothing
    η = T(η)
    NS = length(S)
    S2 = S.^2
    Sinv = @. S/(S2+η)
    F = S2' .- S2
    @. F = F/(CUDAnative.pow(F,2)+η)

    res = ZeroAdder()
    if !(dU isa Nothing)
        J = F.*(U'*dU)
        res += (J+J')*Diagonal(S)
    end

    if !(dV isa Nothing)
        K = F.*(V'*dV)
        res += Diagonal(S) * (K+K')
    end
    if !(dS isa Nothing)
        res += Diagonal(dS)
    end

    res = U*res*V'

    if !(dU isa Nothing) && size(U, 1) != size(U, 2)
        res += (dU - U* (U'*dU)) * Diagonal(Sinv) * V'
    end

    if !(dV isa Nothing) && size(V, 1) != size(V, 2)
        res = res + U * Diagonal(Sinv) * (dV' - (dV'*V)*V')
    end
    res
end

# svd
# svd_back(U::AbstractArray{T}, S, V, dU, dS, dV; η=1e-40) where T
a = randn(ComplexF32, 3,5) |> cu
U, S, V = svd(a)
@test U * Diagonal(S) * V' ≈ a

dU, dS, dV = similar(U), similar(S), similar(V)
svd_back(U, S, V, dU, dS, dV) ≈ svd_back(Matrix(U), Vector(S), Matrix(V), Matrix(dU), Vector(dS), Matrix(dV))
svd_back(U, S, V, dU, dS, dV)

# symeigen
# symeigen_back(E, U, dE, dU; η=1e-40) where T
for T in [Float64, ComplexF32]
    a = randn(ComplexF32, 5,5) |> CuArray; a += a'
    E, U = symeigen(a)
    @test U * Diagonal(E) * U' ≈ a

    dE, dU = similar(E), similar(U)
    @test symeigen_back(E, U, dE, dU) ≈ symeigen_back(Vector(E), Matrix(U), Vector(dE), Matrix(dU))
end

using CuArrays
using GPUArrays
"""
    copyltu!(A::AbstractMatrix) -> AbstractMatrix

copy the lower triangular to upper triangular.
"""
function BackwardsLinalg.copyltu!(A::CuArray)
    m, n = size(A)
    function kernel(arr)
        state = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if state > m*n
            return nothing
        end
        i, j = GPUArrays.gpu_ind2sub(arr, state)
        if i == j
            @inbounds arr[i,i] = real(arr[i,i])
        elseif j > i
            @inbounds arr[i,j] = conj(arr[j,i])
        end
        return nothing
    end
    X = 256
    Y = (m*n-1) ÷ X + 1
    @show X*Y, m*n
    @cuda threads=X blocks=Y kernel(A)
    return A
end


# qr
# qr_back(A, q, r, dq, dr)
a = randn(ComplexF32, 5,9) |> cu
Q, R = qr(a)
@test Q*R ≈ a

dQ, dR = similar(Q), similar(R)
qr_back(a, Q, R, dQ, dR) ≈ qr_back(a, Vector(Q), Matrix(R), Vector(dQ), Matrix(dR))
