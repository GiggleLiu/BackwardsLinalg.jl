using BackwardsLinalg
using LinearAlgebra:svd, Diagonal
using TensorOperations
using Flux, Test

function trg_svd(Ma, Dmax; tol::Float64=1e-12)
    U, S, V = svd(Ma)
    Dmax = min(searchsorted(S, tol, rev=true).stop, Dmax)
    D = isqrt(size(Ma, 1))
    #FS = view(S, 1:Dmax)  # this does not work!!!
    FS = S[1:Dmax]
    S1 = reshape(view(U,:,1:Dmax) .* sqrt.(FS'), (D, D, Dmax))
    S3 = reshape(sqrt.(FS) .* view(V',1:Dmax,:), (Dmax, D, D))
    S1, S3
end

function TRG(K::RT, Dcut::Int, no_iter::Int) where RT
    D = 2
    inds = 1:D

    M = [sqrt(cosh(K)) sqrt(sinh(K));
         sqrt(cosh(K)) -sqrt(sinh(K))]

    T = [mapreduce(a->M[a, i] * M[a, j] * M[a, k] * M[a, l], +, inds) for i in inds, j in inds, k in inds, l in inds]
    eltype(T) <: Tracker.TrackedReal && (T = Tracker.collect(T))

    lnZ = zero(RT)
    for n in 1:no_iter
        maxval = maximum(T)
        T = T/maxval
        lnZ += 2^(no_iter-n+1)*log(maxval)

        D = size(T, 1)

        Ma = reshape(permutedims(T, (3, 2, 1, 4)),  (D^2, D^2))
        Mb = reshape(permutedims(T, (4, 3, 2, 1)),  (D^2, D^2))

        S1, S3 = trg_svd(Ma, Dcut)
        S2, S4 = trg_svd(Mb, Dcut)

        # @tensoropt is much faster than @tensor
        @tensoropt T_new[r, u, l, d] := S1[w, a, r] * S2[a, b, u] * S3[l, b, g] * S4[d, g, w]

        T = T_new
    end
    trace = zero(RT)
    for i in 1:size(T, 1)
        trace += T[i, i, i, i]
    end
    lnZ += log(trace)
end

@testset "trg bp" begin
    Dcut = 24
    n = 3

    K = param(0.5)
    Z = TRG(K, Dcut, n)

    Tracker.back!(Z)
    res = Tracker.grad(K)

    δ = 1e-3
    Z0 = TRG(Tracker.data(K)-δ/2, Dcut, n)
    Z1 = TRG(Tracker.data(K)+δ/2, Dcut, n)
    ngrad = (Z1 - Z0)/δ
    @test isapprox(res, ngrad, atol=1e-2)
end
