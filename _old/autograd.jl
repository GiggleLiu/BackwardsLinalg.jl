using AutoGrad

"""
    svd(A::TrackedArray) -> TrackedTuple

Return tracked tuple of (U, S, V) that `A == USV'`.
"""
@primitive svd!(A), Δ, y svd_back(y..., Δ...)

A = randn(4,4) |> Param
O = randn(4,4)
function test_func(A)
    U, S, V = svd!(A)
    psi = U[:,1]
    psi'*O*psi
end
dU, dS, dV = randn(4,4), randn(4), randn(4,4)
y = @diff test_func(A)
@show value(y)
grad(y, A)

qr(A::TrackedArray) = track(qr, A)
function _forward(::typeof(qr), A)
    Q, R = qr(data(A))
    (Q, R), Δ -> (qr_back(data(A), Q, R, Δ...),)
end

lq(A::TrackedArray) = track(lq, A)
function _forward(::typeof(lq), A)
    L, Q = lq(data(A))
    (L, Q), Δ -> (lq_back(data(A), L, Q, Δ...),)
end
