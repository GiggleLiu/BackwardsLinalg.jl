"""
    svd_back(U, S, V, dU, dS, dV)

adjoint for SVD decomposition.

References:
    https://j-towns.github.io/papers/svd-derivative.pdf
    https://giggleliu.github.io/2019/04/02/einsumbp.html
"""
function svd_back(U::AbstractArray{T}, S, V, dU, dS, dV; η=1e-40) where T
    all(x -> x isa Nothing, (dU, dS, dV)) && return nothing
    NS = length(S)
    S2 = S.^2
    Sinv = @. S/(S2+η)
    F = S2' .- S2
    @. F = F/(F^2+η)

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
