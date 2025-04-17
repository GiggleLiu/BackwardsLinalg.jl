function inv(A::Matrix{T}) where T
    return LinearAlgebra.inv(A)
end

function inv_back(A, B̄)
    B = LinearAlgebra.inv(A)
    Ā = -B' * B̄ * B'
    return Ā
end
