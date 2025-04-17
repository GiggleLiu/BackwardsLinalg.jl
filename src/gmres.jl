function my_gmres(A, b; maxiter = size(A, 2), abstol = 1e-5, reltol = 1e-5, x0 = zeros(length(b)))
	n = length(b)
	x = copy(x0)
	r = b - A * x
	β = norm(r)
	V = zeros(n, maxiter + 1)  # Krylov 子空间基向量
	H0 = zeros(maxiter + 1, maxiter)  # 初始化 Hessenberg 矩阵
	V[:, 1] = r / β  # 第一个基向量

	k = 0  # 记录实际迭代次数
	for j in 1:maxiter
		# Arnoldi 过程
		w = A * V[:, j]
		for i in 1:j
			H0[i, j] = dot(w, V[:, i])
			w -= H0[i, j] * V[:, i]
		end
		H0[j+1, j] = norm(w)
		if H0[j+1, j] < abstol  # 绝对误差判断
			k = j  # 记录实际迭代次数
			break
		end
		V[:, j+1] = w / H0[j+1, j]

		# 最小二乘问题求解
		e1 = zeros(j + 1)
		e1[1] = β
		y = H0[1:j+1, 1:j] \ e1
		x = x0 + V[:, 1:j] * y

		# 相对误差判断
		residual_norm = norm(b - A * x)
		if residual_norm < max(abstol, reltol * norm(b))  # 绝对误差和相对误差的综合判断
			k = j  # 记录实际迭代次数
			break
		end
	end

	# 截取实际使用的 Hessenberg 矩阵
	if k == 0  # 如果未提前退出，则 k = maxiter
		k = maxiter
		println("GMRES did not converge")
	end

	H = H0[1:k+1, 1:k]

	return x, k, H
end

function gmres(A::Matrix{T}, b::Vector{T}; x0 = zeros(T, size(A, 2))) where T <: Number
	if T <: Complex
		n = size(A, 2)
		A1 = [real.(A)  -imag.(A); imag.(A) real.(A)]
		b1 = [real.(b); imag.(b)]
		x1 = my_gmres(A1, b1; x0 = [real.(x0); imag.(x0)])[1]
		return x1[1:n] + im * x1[n+1:2*n]
	end
	return my_gmres(A, b; x0 = x0)[1]
end

function gmres_back(A::Matrix{T}, b::Vector{T}, x̄::Vector; x0 = zeros(T, size(A, 2))) where T <: Number
	
	x = gmres(A,b)
	if LinearAlgebra.norm(A*x-b)<1e-2
		return gmres_back_lneq(A ,b, x, x̄; x0 = x0)
	end

	

	m, n = size(A)
	if T <: Complex
		A1 = [real.(A)  -imag.(A); imag.(A) real.(A)]
		b1 = [real.(b); imag.(b)]
		x0 = [real.(x0); imag.(x0)]
		k = my_gmres(A1, b1; x0 = x0)[2]
	elseif T <: Real
		k = my_gmres(A, b; x0 = x0)[2]
	end
	e1 = zeros(k + 1)
	e1[1] = 1.0
	mask = ones(k + 1, k)
	for j ∈ 1:k
		for i ∈ j+2:k+1
			mask[i, j] = 0.0
		end
	end
	function _gmres(A, b)
		r0 = b - A * x0
		W = hcat([A^(i - 1) * r0 for i in 1:k+1]...)
		Q, R = BackwardsLinalg.qr(W)
		H0 = Q' * A * Q[:, 1:k]
		H = H0 .* mask
		r0e = R[1, 1] * e1
		y = BackwardsLinalg.arg_lstsq(H, r0e)
		x = x0 + Q[:, 1:k] * y
		return x
	end
	if T <: Real
		JA, Jb = Zygote.jacobian(_gmres, A, b)
		Ā = reshape(JA' * x̄, m, n)
		b̄ = Jb' * x̄
	elseif T <: Complex
		JAr, JAi, Jbr, Jbi = Zygote.jacobian((Ar, Ai, br, bi) -> _gmres([Ar -Ai; Ai Ar], [br; bi]), real.(A), imag.(A), real.(b), imag.(b))
		x̄0 = [real.(x̄); imag.(x̄)]
		Ār = reshape(JAr' * x̄0, m, n)
		Āi = reshape(JAi' * x̄0, m, n)
		Ā = Ār + im * Āi
		b̄r = Jbr' * x̄0
		b̄i = Jbi' * x̄0
		b̄ = b̄r + im * b̄i
	end

	return Ā, b̄
end

function gmres_back_lneq(A::Matrix{T}, b::Vector{T}, x::Vector{T}, x̄::Vector; x0 = zeros(T, size(A, 2))) where T <: Number
	b̄ = gmres(Matrix(A'), x̄)
	return -b̄ * x', b̄
end

