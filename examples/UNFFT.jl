using NFFT, LinearAlgebra, Random, BackwardsLinalg, IterativeSolvers, NFFTTools

Random.seed!(3)
J = N = 128;
k = range(-0.4, stop=0.4, length=J);  
f = randn(ComplexF64, J);
p = plan_nfft(k, N, reltol=1e-9);
A = BackwardsLinalg.A_construct_t2(k);
fhat = p*f;
f1 = A^(-1)*fhat;
norm(A*f1 - fhat)
W = sdc(p, iters = 10);
B = A'*diagm(W)*A;
b = A'*diagm(W)*fhat;
f2 = B\b;
norm(A*f2 - fhat)
B

A*diagm(W)*A'

g1 = A'^(-1)*f
norm(A*g1 - f)
C = A*diagm(W)*A'
c = A*diagm(W)*fhat
g2 = C\c
norm(A*g2 - fhat)

f3 = gmres(B,b; reltol=1e-8, abstol=1e-8, verbose=true)
norm(A*f3 - fhat)

sdc(p,iters = 10)


###########
Random.seed!(3)
N = 128
k = rand(N) .- 0.5
A = BackwardsLinalg.A_construct_t2(k)
f = randn(ComplexF64, N)
fhat = NFFT.nfft(k,f)


f1 = A^(-1)*fhat
error1 = norm(A*f1 - fhat)

f2 = gmres(A'*A, A'*fhat;reltol=1e-8, abstol=1e-8, verbose=true)

error2 = norm(A*f2 - fhat)



A = rand(128,128) + 128*I
b = rand(128)

x = gmres(A, b; reltol=1e-8, abstol=1e-8, verbose=true)


norm(A*x - b)