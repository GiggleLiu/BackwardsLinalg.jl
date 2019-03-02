using LinearAlgebra
import LinearAlgebra: qr

export _qr

# the one in tensorflow package
function qr_back(q, r, dq, dr)
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

function copyltu!(A::AbstractMatrix)
    m, n = size(A)
    for i=1:m-1
        for j=i+1:n
            @inbounds A[i,j] = A[j,i]
        end
    end
    A
end

# the one in the paper
function qr_back(q, r, dq, dr)
    dq isa Nothing && (dq = zero(q))  # fix this lazy impl
    dr isa Nothing && (dr = zero(r))
    M = r*dr' - dq'*q
    b = dq + q*copyltu!(M)
    #(dq + q*copyltu!(M))*inv(r')
    LAPACK.trtrs!('U', 'N', 'N', r, Matrix(b'))'
end

function qr_back_rankdef(A, q, r, dq, dr)
    size(r, 1) == size(r, 2) && return qr_back(q, r, dq ,dr)
    dq isa Nothing && (dq = zero(q))  # fix this lazy impl
    dr isa Nothing && (dr = zero(r))
    M, N = size(r)
    B = view(A,:,M+1:N)
    dU = view(dr,:,1:M)
    dD = view(dr,:,M+1:N)
    U = view(r,:,1:M)
    D = view(r,:,M+1:N)
    da = qr_back(q, U, dq+B*dD', dU)
    db = q*dD
    hcat(da, db)
end

function _qr(x)
    res = qr(x)
    Matrix(res.Q), res.R
end

#using AutoGrad
#@primitive _qr(x),dy,y qr_back_rankdef(x, y..., dy...)

_qr(A::TrackedArray) = track(_qr, A)
function _forward(::typeof(_qr), A)
    Q, R = _qr(data(A))
    (Q, R), Δ -> (qr_back_rankdef(data(A), Q, R, Δ...),)
end
