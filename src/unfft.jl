function unfft(k,f)
    return NFFT.nfft(k,f)
end

function unfft_back(k,ȳ)
    return NFFT.nfft_adjoint(k,length(ȳ),ȳ)
end


# Type 2: f̂ⱼ = ∑ₙ fₙ exp(-2πi n kⱼ) n ∈ [-N/2, N/2)
function A_construct_t2(k)
    n = length(k)
    A = zeros(ComplexF64, n, n)
    IN = collect(- n>>1:1: n>>1-1)
    for i in 1:n
        for j in 1:n
            A[i,j] = exp(- 2π*im*k[i]*IN[j])
        end
    end
    return A
end


function inufft_t2(k,fhat;iters = 10)
    A = A_construct_t2(k)
    p = plan_nfft(k, length(k))
    W = NFFTTools.sdc(p, iters = iters)
    B=  A'*LinearAlgebra.diagm(W)*A
    b = A'*LinearAlgebra.diagm(W)*fhat
    res = B\b
    @show LinearAlgebra.norm(A*res - fhat)
    return res
end

#=
function inufft_t2(k,fhat)
    return A_construct_t2(k)\fhat
end
=#


function inufft_t2_back(k,f̄;iters = 10)
    A = A_construct_t2(k)
    #=
    p = plan_nfft(k, length(k))
    W = NFFTTools.sdc(p, iters = iters)
    B = A*LinearAlgebra.diagm(W)*A'
    b = A*LinearAlgebra.diagm(W)*f̄
    return B\b
    =#
    res = A'\f̄ 
    @show LinearAlgebra.norm(A'*res - f̄)
    return res
end


