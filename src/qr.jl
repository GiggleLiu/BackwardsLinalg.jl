using LinearAlgebra, Flux
using Flux.Tracker: @grad, data, track, TrackedTuple, TrackedArray
import Flux.Tracker: _forward
import LinearAlgebra: qr

export _qr, qr_back

#=
# the one in tensorflow package
function qr_back_fullrank(q, r, dq, dr)
    size(r, 1) != size(r, 2) && throw(NotImplementedError("QrGrad not implemented when ncols > nrows or full_matrices is true and ncols != nrows."))
    dq isa Nothing && (dq = zero(q))
    dr isa Nothing && (dr = zero(r))

    qdq = q' * dq
    qdq_ = qdq - qdq'
    rdr = r * dr'
    rdr_ = rdr - rdr'
    ut = tril!(qdq_ + rdr_)

    function trsolve(r, x)
        LAPACK.trtrs!('U', 'N', 'N', r, Matrix(x'))'
    end

    grad_a = q * (dr + trsolve(r, ut))
    grad_b = trsolve(r, dq - q * qdq)
    grad_a + grad_b
end
=#

"""
    copyltu!(A::AbstractMatrix) -> AbstractMatrix

copy the lower triangular to upper triangular.
"""
function copyltu!(A::AbstractMatrix)
    m, n = size(A)
    for i=1:m-1
        for j=i+1:n
            @inbounds A[i,j] = A[j,i]
        end
    end
    A
end

using MacroTools
"""
    qr_back_fullrank(q, r, dq, dr) -> Matrix

backward for QR decomposition, for input matrix (in forward pass) with M > N.

References:
    Seeger, M., Hetzel, A., Dai, Z., Meissner, E., & Lawrence, N. D. (2018). Auto-Differentiating Linear Algebra.
"""
@generated function qr_back_fullrank(q, r, dq, dr)
    dqnot0 = !(dq <: Nothing)
    drnot0 = !(dr <: Nothing)
    ex = drnot0 ? :(r*dr') : :()
    ex = dqnot0 ? :($ex - dq'*q) : ex
    :(b = $(dqnot0 ? :(dq) : :()) + q*copyltu!($ex);
      LAPACK.trtrs!('U', 'N', 'N', r, Matrix(b'))')
end

"""
    qr_back(A, q, r, dq, dr) -> Matrix

backward for QR decomposition, for an arbituary shaped input matrix.

References:
    Seeger, M., Hetzel, A., Dai, Z., Meissner, E., & Lawrence, N. D. (2018). Auto-Differentiating Linear Algebra.
    HaiJun's paper.
"""
@generated function qr_back(A, q, r, dq, dr)
    dqnot0 = !(dq <: Nothing)
    drnot0 = !(dr <: Nothing)
    ex = quote
        size(r, 1) == size(r, 2) && return qr_back_fullrank(q, r, dq ,dr)
        M, N = size(r)
        B = view(A,:,M+1:N)
        U = view(r,:,1:M)
        D = view(r,:,M+1:N)
        $(if drnot0
            :(dD = view(dr,:,M+1:N);
            da = qr_back_fullrank(q, U, $(dqnot0 ? :(dq+B*dD') : :(B*dD')), view(dr,:,1:M));
            db = q*dD)
        else
            :(da = qr_back_fullrank(q, U, dq, nothing);
            db = zero(B))
        end)
        hcat(da, db)
    end
end

function _qr(x)
    res = qr(x)
    Matrix(res.Q), res.R
end

_qr(A::TrackedArray) = track(_qr, A)
function _forward(::typeof(_qr), A)
    Q, R = _qr(data(A))
    (Q, R), Δ -> (qr_back(data(A), Q, R, Δ...),)
end
