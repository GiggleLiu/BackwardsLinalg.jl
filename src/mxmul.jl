function mxmul(A::Matrix{T},B::Matrix{T}) where T
    return A*B
end

function mxmul_back(A::Matrix{T}, B::Matrix{T}, C̄::Matrix{T}) where T
    return C̄*B', A'*C̄
end
