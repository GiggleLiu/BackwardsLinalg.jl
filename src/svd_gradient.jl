using LinearAlgebra

"""
    svd_back(U, S, V, dU, dS, dV)

adjoint for SVD decomposition.

References:
    https://j-towns.github.io/papers/svd-derivative.pdf
    https://giggleliu.github.io/2019/04/02/einsumbp.html
"""
function svd_back(U::AbstractArray{T}, S, V, dU, dS, dV; η=1e-12) where T
    all(isnothing, (dU, dS, dV)) && return nothing
    NS = length(S)
    S2 = S.^2
    Sinv = @. S/(S2+η)
    F = S2' .- S2
    @. F = F/(F^2+η)

    res = ZeroAdder()
    if !isnothing(dU)
        J = F.*(U'*dU)
        res += (J+J')*Diagonal(S)
    end

    if !isnothing(dV)
        K = F.*(V'*dV)
        res += Diagonal(S) * (K+K')
    end
    if !isnothing(dS)
        res += Diagonal(dS)
    end

    res = U*res*V'

    if !isnothing(dU) && size(U, 1) != size(U, 2)
        res += (dU - U* (U'*dU)) * Diagonal(Sinv) * V'
    end

    if !isnothing(dV) && size(V, 1) != size(V, 2)
        res = res + U * Diagonal(Sinv) * (dV' - (dV'*V)*V')
    end
    res
end
