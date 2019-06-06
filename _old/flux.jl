using Flux.Tracker: @grad, data, track, TrackedTuple, TrackedArray
import Flux.Tracker: _forward
Base.iterate(xs::TrackedTuple, state=1) = state > length(xs) ? nothing : (xs[state], state+1)

"""
    svd(A::TrackedArray) -> TrackedTuple

Return tracked tuple of (U, S, V) that `A == USV'`.
"""
svd!(A::TrackedArray) = track(svd!, A)
function _forward(::typeof(svd!), a)
    U, S, V = svd!(data(a))
    (U, S, V), Δ -> (svd_back(U, S, V, Δ...),)
end

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
