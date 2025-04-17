using LinearAlgebra

function lu(A::Matrix{T}) where T
    m, n = size(A)
    if m != n
        error("LU 分解仅适用于方阵")
    end

    # 初始化 L, U, P
    L = Matrix{T}(I, m, m)  # 单位下三角矩阵
    U = copy(A)             # 上三角矩阵
    P = Matrix{T}(I, m, m)  # 置换矩阵

    for k in 1:n-1
        # 部分选主元：找到第 k 列中绝对值最大的元素
        pivot_row = argmax(abs.(U[k:end, k])) + k - 1

        # 交换行
        if pivot_row != k
            U[[k, pivot_row], :] = U[[pivot_row, k], :]
            P[[k, pivot_row], :] = P[[pivot_row, k], :]
            if k > 1
                L[[k, pivot_row], 1:k-1] = L[[pivot_row, k], 1:k-1]
            end
        end

        # 高斯消元
        for i in k+1:n
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:end] -= L[i, k] * U[k, k:end]
        end
    end

    return L, U, P
end


function lu_back(A, L̄0, Ū0, P̄)
	L,U,P = lu(A)
	n = size(A, 1)
	K = ones(n, n)
	K = LinearAlgebra.UpperTriangular(K)
	J = ones(n, n) - K
    L̄ = L̄0 .* J
    Ū = Ū0 .* K
	Ā = P * (L')^(-1) * ((Ū * U') .* K + (L' * L̄) .* J) * (U')^(-1)
	return Ā
end

