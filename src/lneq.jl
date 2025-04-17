function lneq(A::Matrix{T}, b::Vector{T}) where T <: Number
	return A \ b
end


function lneq_back(A::Matrix{T}, b::Vector{T}, x, x̄) where T
	b̄ = (A')^(-1) * x̄
	Ā = - (A')^(-1) * x̄ * x'
	return Ā, b̄
end


