export qr_back, copyltu!, lq, lq_back

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
        LinearAlgebra.LAPACK.trtrs!('U', 'N', 'N', r, Matrix(x'))'
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

"""
    qr_back_fullrank(q, r, dq, dr) -> Matrix

backward for QR decomposition, for input matrix (in forward pass) with M > N.

References:
    Seeger, M., Hetzel, A., Dai, Z., Meissner, E., & Lawrence, N. D. (2018). Auto-Differentiating Linear Algebra.
"""
@generated function qr_back_fullrank(q, r, dq, dr)
    dqnot0 = !(dq <: Nothing)
    drnot0 = !(dr <: Nothing)
    (!dqnot0 && !drnot0) && return :(nothing)
    ex = drnot0 && dqnot0 ? :(r*dr' - dq'*q) : (dqnot0 ? :(-dq'*q) : :(r*dr'))
    :(b = $(dqnot0 ? :(dq) : :(ZeroAdder())) + q*copyltu!($ex);
      LinearAlgebra.LAPACK.trtrs!('U', 'N', 'N', r, Matrix(b'))')
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
    (!dqnot0 && !drnot0) && return :(nothing)
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

"""
    lq_back(A, l, q, dl, dq) -> Matrix

backward for LQ decomposition, for an arbituary shaped input matrix.

References:
    Seeger, M., Hetzel, A., Dai, Z., Meissner, E., & Lawrence, N. D. (2018). Auto-Differentiating Linear Algebra.
    HaiJun's paper.
"""
function lq_back(A, l, q, dl, dq)
    qr_back(A', q', l' |> Matrix, dq', dl')' |> Matrix
end
