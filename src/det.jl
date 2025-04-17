function cofactor_matrix(A::Matrix{T}) where T
    n = size(A, 1)
    C = zeros(T, n, n)  # 初始化代数余子式矩阵
    for i in 1:n
        for j in 1:n
            # 计算余子式
            minor = A[setdiff(1:n, i), setdiff(1:n, j)]
            C[i, j] = (-1)^(i+j) * det(minor)
        end
    end
    return C
end

# 计算伴随矩阵
function adjugate_matrix(A::Matrix{T}) where T
    return transpose(cofactor_matrix(A))  # 代数余子式矩阵的转置
end


function det(A::Matrix{T}) where T
    return LinearAlgebra.det(A)
end



function det_back(A,ā)
    detA = LinearAlgebra.det(A)
    if norm(detA)> 1e-12 
        Aad = A^(-1)*LinearAlgebra.det(A)
    else
        Aad = adjugate_matrix(A)
    end
    return ā*Aad'
end


