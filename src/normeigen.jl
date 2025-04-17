function normeigen(A::AbstractMatrix)
	E, U = LinearAlgebra.eigen(A)
	E .+ 0.0im, Matrix(U)
end



function normeigen_back(E::AbstractVector{T}, U, dE, dU; η=1e-40) where T
	all(x->x isa AbstractZero, (dU, dE)) && return NoTangent()
	η = T(η)
	if dU isa AbstractZero
		D = LinearAlgebra.Diagonal(dE)
	else
		F = -(E .- transpose(E))
		F .= F./(F.^2 .+ η)
		D = 1/2 * (U' * dU - dU'*U) .* conj.(F)
		if !(dE isa AbstractZero)
			D = D + LinearAlgebra.Diagonal(dE)
		end
	end
	return U * D * U'
end









