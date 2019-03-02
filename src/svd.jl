import LinearAlgebra: svd!, svd
using LinearAlgebra
using Flux.Tracker: @grad, data, track, TrackedTuple, TrackedArray
import Flux.Tracker: _forward

export _svd, _svd!, svd_back

"""
    svd_back(U, S, V, dU, dS, dV)

backward for SVD decomposition

References:
    https://j-towns.github.io/papers/svd-derivative.pdf
"""
function svd_back(U, S, V, dU, dS, dV; η=1e-12)
    NS = length(S)
    S2 = S.^2
    Sinv = @. S/(S2+η)
    F = S2' .- S2
    @. F = F/(F^2+η)

    UdU = U'*dU
    VdV = V'*dV

    Su = (F.*(UdU-UdU'))*LinearAlgebra.Diagonal(S)
    Sv = LinearAlgebra.Diagonal(S) * (F.*(VdV-VdV'))

    U * (Su + Sv + LinearAlgebra.Diagonal(dS)) * V' +
    (LinearAlgebra.I - U*U') * dU*LinearAlgebra.Diagonal(Sinv) * V' +
    U*LinearAlgebra.Diagonal(Sinv) * dV' * (LinearAlgebra.I - V*V')
end

function _svd!(A)
    U, S, V = svd!(A)
    U, S, Matrix(V)
end

"""
    _svd(A::TrackedArray) -> TrackedTuple

Return tracked tuple of (U, S, V) that `A == USV'`.
"""
_svd!(A::TrackedArray) = track(_svd!, A)
_svd(A) = _svd!(A |> copy)
function _forward(::typeof(_svd!), a)
    U, S, V = _svd!(data(a))
    (U, S, V), Δ -> (svd_back(U, S, V, Δ...),)
end

Base.iterate(xs::TrackedTuple, state=1) = state > length(xs) ? nothing : (xs[state], state+1)
