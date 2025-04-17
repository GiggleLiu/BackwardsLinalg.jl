function pf(A::Matrix{T}) where T<:Number
    return pfaffian(A)
end

function pf_back(A::Matrix{T}, pfA, ā) where T<: Number
    Aad = adjugate_matrix(A)
    Ā = - ā * Aad / (2 * pfA)
    return (Ā - Ā')/2
end