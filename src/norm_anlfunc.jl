function norm_anlfunc(f, df, A::Matrix{T}) where T
	S, U = LinearAlgebra.eigen(A)
	return U * diagm(f.(S)) * U'
end

function norm_anlfunc_back(f, df, A::Matrix{T}, B̄) where T
	S, U = LinearAlgebra.eigen(A)
	fs = diagm(f.(S))
	Ū = B̄ * U * fs' + B̄' * U * fs
	n = size(A, 1)
	S̄0 = (diagm(df.(S))' * U' * B̄ * U) .* LinearAlgebra.I(n)
	S̄ = diag(S̄0)
	Ā = normeigen_back(S, U, S̄, Ū)
	return Ā
end


