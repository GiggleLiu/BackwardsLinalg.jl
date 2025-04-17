function symeigen(A::AbstractMatrix)
	E, U = LinearAlgebra.eigen(Hermitian(A))
	E, Matrix(U)
end

"""
References:
	* Seeger, M., Hetzel, A., Dai, Z., Meissner, E., & Lawrence, N. D. (2018). Auto-Differentiating Linear Algebra.
"""



function symeigen_back(E::AbstractVector{T}, U, dE, dU; η=1e-40) where T
    all(x->x isa AbstractZero, (dU, dE)) && return NoTangent()
    η = T(η)
    if dU isa AbstractZero
        D = LinearAlgebra.Diagonal(dE)
    else
        F = -(E .- E')
        F .= F./(F.^2 .+ η)
        dUU = (U' * dU) .* F
        D = (dUU + dUU')/2
        if !(dE isa AbstractZero)
            D = D + LinearAlgebra.Diagonal(dE)
        end
    end
    U * D * U'
end







