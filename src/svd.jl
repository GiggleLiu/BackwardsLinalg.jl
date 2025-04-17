mpow2(a::AbstractArray) = a .^ 2

"""
		svd(A) -> Tuple{AbstractMatrix, AbstractVector, AbstractMatrix}
"""
function svd(A)
	U, S, V = LinearAlgebra.svd(A)
	return U, S, Matrix(V)
end

"""
    svd_back(U, S, V, dU, dS, dV)

adjoint for SVD decomposition.

References:
    https://j-towns.github.io/papers/svd-derivative.pdf
    https://giggleliu.github.io/2019/04/02/einsumbp.html
"""
# Here input S and dS are both vector
function svd_back(U::AbstractArray, S::AbstractArray{T}, V, dU, dS, dV; η::Real=1e-40) where T
    all(x -> x isa Nothing, (dU, dS, dV)) && return nothing
    η = T(η)
    NS = length(S)
    S2 = mpow2(S)
    Sinv = @. S/(S2+η)
    F = S2' .- S2
    F ./= (mpow2(F) .+ η)

    res = ZeroAdder()
    if !(dU isa AbstractZero)
        UdU = U'*dU
        J = F.*(UdU)
        res += (J+J')*LinearAlgebra.Diagonal(S) + LinearAlgebra.Diagonal(1im*imag(LinearAlgebra.diag(UdU)) .* Sinv)
    end
    if !(dV isa AbstractZero)
        VdV = V'*dV
        K = F.*(VdV)
        res += LinearAlgebra.Diagonal(S) * (K+K')
    end
    if !(dS isa AbstractZero)
        res += LinearAlgebra.Diagonal(dS)
    end

    res = U*res*V'

    if !(dU isa AbstractZero) && size(U, 1) != size(U, 2)
        res += (dU - U* (U'*dU)) * LinearAlgebra.Diagonal(Sinv) * V'
    end

    if !(dV isa AbstractZero) && size(V, 1) != size(V, 2)
        res = res + U * LinearAlgebra.Diagonal(Sinv) * (dV' - (dV'*V)*V')
    end
    res
end
