using LinearAlgebra
import LinearAlgebra: qr
using AutoGrad

function qr_back(q, r, dq, dr)
    size(r, 1) != size(r, 2) && throw(NotImplementedError("QrGrad not implemented when ncols > nrows or full_matrices is true and ncols != nrows."))
    dq isa Nothing && (dq = zero(q))
    dr isa Nothing && (dr = zero(r))

    qdq = q' * dq
    qdq_ = qdq - qdq'
    rdr = r * dr'
    rdr_ = rdr - rdr'
    ut = tril(qdq_ + rdr_)

    trsolve(x, r) = LAPACK.trtrs!('U', 'N', 'N', x' |> Matrix, r)[1]'

    grad_a = q * (dr + trsolve(ut, r))
    grad_b = trsolve(dq - q * qdq, r)
    grad_a + grad_b
end

function _qr(x)
    res = qr(x)
    Matrix(res.Q), res.R
end
@primitive _qr(x),dy,y qr_back(y..., dy...)

x = Param(randn(4,4))
b = randn(4)
y = @diff _qr(x)[2] |> sum
grad(y, x)

using Test
@testset "qr" begin
    M, N = 4, 4
    A = randn(M, N)
    b = randn(M, N)
    AutoGrad.gradcheck(x->_qr(x)[2]*b |> sum, A) |> println
    print("Test Pass!")
end
