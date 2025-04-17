using Zygote, LinearAlgebra, BackwardsLinalg, Random


T = Float64
Random.seed!(3)
n = 200
A = rand(T, n, n) + n/64 * I
b = rand(T, n)

x= BackwardsLinalg.gmres(A, b)
norm(A*x-b)
x̄ = rand(T ,n)
BackwardsLinalg.my_gmres(A,b)[2]

# ========


e1 = zeros(T, k + 1)
e1[1] = 1.0
m, n = size(A)
mask = ones(T, k + 1, k)
for j ∈ 1:k
	for i ∈ j+2:k+1
		mask[i, j] = 0.0
	end
end


x0 = zeros(n)
r0 = b - A * x0
W = hcat([A^(i - 1) * r0 for i in 1:k+1]...)
Q,R = BackwardsLinalg.qr(W)

H0 = Q' * A * Q[:, 1:k]
H1 = H0 .* mask
r0e = R[1,1] * e1
y = BackwardsLinalg.arg_lstsq(H1, r0e)
x1 = x0 + Q[:, 1:k] * y



norm(A*x1-b)
norm(x1-x)/norm(x)



# --------------------

x0 = zeros(n)
r0 = b - A * x0
W = hcat([A^(i - 1) * r0 for i in 1:k+1]...)
Q, R = BackwardsLinalg.qr(W)
Q = Q[:,1:k+1]
β = R[1,1]
H0 = R[1:k+1,2:k+1]

r0e = β * e1
y = BackwardsLinalg.arg_lstsq(H0, r0e)
x1 = x0 + Q[:, 1:k] * y
x

norm(A*x1-b)
norm(x1-x)/norm(x)

function tf(A)
	B = copy(A)
	B = 2*B
	return sum(abs2.(B))
end

A =rand(3,3)

tf(A)
gradient(tf,A)

A = rand(100,5)
res = LinearAlgebra.qr(A)
Matrix(res.Q)