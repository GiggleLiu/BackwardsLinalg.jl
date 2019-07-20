using Zygote
using Zygote:@adjoint
using Random

export gradient_check

@adjoint function qr(A)
    res = qr(A)
    Q, R = Matrix(res.Q), res.R
    (Q, R), dy -> (qr_back(A, Q, R, dy...),)
end

@adjoint function qr(A::AbstractMatrix, pivot::Val{true})
    res = qr(A, pivot)
    Q, R, P = Matrix(res.Q), res.R, res.P
    (Q, R, P), dy -> (qr_back(Q*R, Q, R, dy[1], dy[2])*P',nothing)
end

@adjoint function lq(A)
    res = lq(A)
    L, Q = res.L, Matrix(res.Q)
    (L, Q), dy -> (lq_back(A, L, Q, dy...),)
end

@adjoint function svd(A)
    U, S, V = svd(A)
    V = Matrix(V)
    (U, S, V), dy -> (svd_back(U, S, V, dy...),)
end

@adjoint function rsvd(A, args...; kwargs...)
    U, S, V = rsvd(A, args...; kwargs...)
    (U, S, V), dy -> (svd_back(U, S, V, dy...),)
end

@adjoint function symeigen(A)
    E, U = eigen(A)
    U = Matrix(U)
    (E, U), adjy -> (symeigen_back(E, U, adjy...),)
end

function gradient_check(f, args...; η = 1e-5)
    g = gradient(f, args...)
    dy_expect = η*sum(abs2.(g[1]))
    dy = f(args...)-f([gi == nothing ? arg : arg.-η.*gi for (arg, gi) in zip(args, g)]...)
    @show dy
    @show dy_expect
    isapprox(dy, dy_expect, rtol=1e-2, atol=1e-8)
end

@adjoint Random.seed!(n) = Random.seed!(n), _ -> nothing
