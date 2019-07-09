function symeigen(A::AbstractMatrix)
    E, U = eigen(A)
    E, Matrix(U)
end

"""
References:
    * Seeger, M., Hetzel, A., Dai, Z., Meissner, E., & Lawrence, N. D. (2018). Auto-Differentiating Linear Algebra.
"""
function symeigen_back(E::AbstractVector{T}, U, dE, dU; η=1e-40) where T
    all(x->x isa Nothing, (dU, dE)) && return nothing
    η = T(η)
    if dU === nothing
        D = Diagonal(dE)
    else
        F = E .- E'
        F .= F./(F.^2 .+ η)
        dUU = dU' * U .* F
        D = (dUU + dUU')/2
        if dE !== nothing
            D = D + Diagonal(dE)
        end
    end
    U * D * U'
end
