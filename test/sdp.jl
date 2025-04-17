using JuMP, SCS, LinearAlgebra, Random
using Test, Zygote, BackwardsLinalg


function gradient_check(f, args...; η = 1e-5)
	g = gradient(f, args...)
	dy_expect = η * sum(abs2.(g[1]))
	@show dy_expect
	dy = f(args...) - f([gi === nothing ? arg : arg .- η .* gi for (arg, gi) in zip(args, g)]...)
	@show dy
	isapprox(dy, dy_expect, rtol = 1e-2, atol = 1e-8)
end

function gradient_check_A(f, A, η = 1e-5)
	Ā = gradient(f, A)[1]
	dy_expect = η * sum(sum(map(Ai -> abs2.(Ai), Ā)))
	@show dy_expect
	dy = f(A) - f(A .- η .* Ā)
	@show dy
	isapprox(dy, dy_expect, rtol = 5*1e-2, atol = 1e-8)
end


@testset "sdp grad for b" begin
	# 定义数据
	Random.seed!(123)  # 设置随机种子
	n = 4  # 矩阵的维度

	# 生成对称的目标矩阵 C
	C = rand(n, n)
	C[2, 3] += 0.1
	C = (C + C') / 2  # 确保对称性

	# 生成对称的约束矩阵 A1 和 A2
	A1 = rand(n, n)
	A1 = (A1 + A1') / 2  # 确保对称性

	A2 = rand(n, n)
	A2 = (A2 + A2') / 2  # 确保对称性

	# 生成约束的右侧值 b1 和 b2
	b1 = tr(A1 * I(n))  # 约束 1 的右侧值 b1，确保可行
	b2 = tr(A2 * I(n))  # 约束 2 的右侧值 b2，确保可行

	A = [A1, A2]
	b = [b1, b2]

	testfb(b) = tr(BackwardsLinalg.sdp(C, A, b))
	testfC(C) = tr(BackwardsLinalg.sdp(C, A, b))
	testfA(A) = tr(BackwardsLinalg.sdp(C, A, b))

	@test gradient_check(testfb, b)
	@test gradient_check(testfC, C)
	@test gradient_check_A(testfA, A)
end








