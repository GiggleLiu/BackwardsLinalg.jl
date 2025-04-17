using BackwardsLinalg, NFFT, LinearAlgebra,Random
using Plots,Zygote

Random.seed!(3)
N = 16
k = rand(N) .- 0.5
f = randn(ComplexF64, N)
A = BackwardsLinalg.A_construct_t2(k)
cond(A)
loss(x)=sum(abs2.(A*x))
η = 1e-5
step = 1000

J = zeros(step)
for i in 1:step
    J[i] = norm(gradient(loss,f)[1])
    f = f - η*gradient(loss,f)[1]
end
plot(J)


using LinearAlgebra,Plots,Random
Random.seed!(3)
M = 20
cond_num = zeros(M)
T = 1000000
for i in 1: M
    cond0 = 0.0
    for t in 1:T
        if t%1000 == 0
            println(i)
        end
        A =  rand(i,i)
        cond0 += cond(A)
    end
    cond_num[i] = cond0/T
end
plot(1:M,cond_num)



cond_num[2]