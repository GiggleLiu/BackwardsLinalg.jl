function cls(A::Matrix{T}) where T
    L = Matrix(cholesky(A).L)
    return L
end
    

function cls_back(A::Matrix{T}, L̄) where T
    L = BackwardsLinalg.cls(A)
    n = size(A)[1]
    M = ones(T, n, n)
    M[diagind(A)] .= 0.5
    M = LinearAlgebra.UpperTriangular(M)
    Ā = 0.5 * (L')^(-1) * ( (L'*L̄).*M' + (L̄'*L).*M )*L^(-1)
    Ā = (Ā' + Ā)/2
    return Ā
end

