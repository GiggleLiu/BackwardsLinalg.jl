using Flux.Tracker: @grad, data, track, TrackedTuple, TrackedArray
import Flux.Tracker: _forward
Base.iterate(xs::TrackedTuple, state=1) = state > length(xs) ? nothing : (xs[state], state+1)

"""
    _svd(A::TrackedArray) -> TrackedTuple

Return tracked tuple of (U, S, V) that `A == USV'`.
"""
_svd!(A::TrackedArray) = track(_svd!, A)
function _forward(::typeof(_svd!), a)
    U, S, V = _svd!(data(a))
    (U, S, V), Δ -> (svd_back(U, S, V, Δ...),)
end

_qr(A::TrackedArray) = track(_qr, A)
function _forward(::typeof(_qr), A)
    Q, R = _qr(data(A))
    (Q, R), Δ -> (qr_back(data(A), Q, R, Δ...),)
end

_diageig(A::TrackedArray) = track(_diageig, A)
function _forward(::typeof(_diageig), A)
    λ, U = _diageig(data(A))
    (λ, U), Δ -> (diageig_back(λ, U, Δ...),)
end