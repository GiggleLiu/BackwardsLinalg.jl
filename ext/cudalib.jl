using CuArrays
using CuArrays: CUSOLVER, CUBLAS, CURAND
using CUDAnative
import CUDAnative: pow, abs, angle
using GPUArrays

cuseed!(seed::Int) = CURAND.set_pseudo_random_generator_seed(CURAND.generator(), seed)
function CURAND.curandn(::Type{T}, size...) where {FT, T<:Complex{FT}}
    return curandn(FT, size...) + im*curandn(FT, size...)
end
do_adjoint(A::CuMatrix) = CuMatrix(A')

for (RT, CT) in [(:Float64, :ComplexF64), (:Float32, :ComplexF32)]
    @eval CUDAnative.angle(z::$CT) = CUDAnative.atan2(CUDAnative.imag(z), CUDAnative.real(z))
    @eval function CUDAnative.abs(z::$CT)
        i = CUDAnative.imag(z)
        r = CUDAnative.real(z)
        CUDAnative.sqrt(i*i+r*r)
    end

    @eval cp2c(d::$RT, a::$RT) = CUDAnative.ComplexF64(d*CUDAnative.cos(a), d*CUDAnative.sin(a))
    for NT in [RT, :Int32, :Int64]
        @eval CUDAnative.pow(z::$CT, n::$NT) = CUDAnative.$CT((CUDAnative.pow(CUDAnative.abs(z), n)*CUDAnative.cos(n*CUDAnative.angle(z))), (CUDAnative.pow(CUDAnative.abs(z), n)*CUDAnative.sin(n*CUDAnative.angle(z))))
    end
end

mpow2(a::CuArray) = CUDAnative.pow.(a, 2)
trtrs!(c1::Char, c2::Char, c3::Char, r::CuArray, b::CuVector) = CUBLAS.trsv!(c1, c2, c3, r, b)
trtrs!(c1::Char, c2::Char, c3::Char, r::CuArray{T}, b::CuMatrix{T}) where T = CUBLAS.trsm!('L',c1, c2, c3, T(1), r, b)

function symeigen(a::CuArray{<:Complex})
    CUSOLVER.heevd!('V','U',copy(a))
end

function symeigen(a::CuArray{<:Real})
    CUSOLVER.syevd!('V','U',copy(a))
end

function qr(a::CuArray)
    M, N = size(a)
    A, tau = CUSOLVER.geqrf!(copy(a))
    if M > N
        R = CuArrays.triu(A[1:N,:])
    else
        R = CuArrays.triu(A)
    end
    Q = CUSOLVER.orgqr!(A, tau)
    return Q, R
end

"""
    copyltu!(A::AbstractMatrix) -> AbstractMatrix

copy the lower triangular to upper triangular.
"""
function copyltu!(A::CuArray)
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

    X, Y = GPUArrays.thread_blocks_heuristic(m*n)
    @cuda blocks=X threads=Y kernel(A)
    return A
end
