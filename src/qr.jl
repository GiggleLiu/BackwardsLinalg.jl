"""
		qr(A) -> Tuple{AbstractMatrix, AbstractMatrix}

private QR method, call LinearAlgebra.qr to achieve forward calculation, while
return Tuple type
"""
function qr(A)
	res = LinearAlgebra.qr(A)
	Matrix(res.Q), res.R
end

"""
		qr(A, pivot) -> Tuple{AbstractMatrix, AbstractMatrix, AbstractVector}
"""
function qr(A::AbstractMatrix, pivot::Val{true})
    res = LinearAlgebra.qr(A, pivot)
    Q, R, P = Matrix(res.Q), res.R, res.P
end

"""
		lq(A) -> Tuple{AbstractMatrix, AbstractMatrix}
"""
function lq(A)
		res = LinearAlgebra.lq(A)
    res.L, Matrix(res.Q)
end


trtrs!(c1::Char, c2::Char, c3::Char, r::AbstractMatrix, b::AbstractVecOrMat) = LinearAlgebra.LAPACK.trtrs!(c1, c2, c3, r, b)

"""
    copyltu!(A::AbstractMatrix) -> AbstractMatrix

copy the lower triangular to upper triangular.
"""
function copyltu!(A::AbstractMatrix)
    m, n = size(A)
    for i=1:m
        A[i,i] = real(A[i,i])
        for j=i+1:n
            @inbounds A[i,j] = conj(A[j,i])
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
function qr_back_fullrank(q, r, dq, dr)
    dqnot0 = !(dq isa AbstractZero)
    drnot0 = !(dr isa AbstractZero)
    (!dqnot0 && !drnot0) && return NoTangent()
    ex = drnot0 && dqnot0 ? r*dr' - dq'*q : (dqnot0 ? -dq'*q : r*dr')
    b = dqnot0 ? dq + q*copyltu!(ex) : q*copyltu!(ex)
    trtrs!('U', 'N', 'N', r, do_adjoint(b))'
end

do_adjoint(A::Matrix) = Matrix(A')

"""
    qr_back(A, q, r, dq, dr) -> Matrix

backward for QR decomposition, for an arbituary shaped input matrix.

References:
    Seeger, M., Hetzel, A., Dai, Z., Meissner, E., & Lawrence, N. D. (2018). Auto-Differentiating Linear Algebra.
    Differentiable Programming Tensor Networks, Hai-Jun Liao, Jin-Guo Liu, Lei Wang, Tao Xiang
"""
function qr_back(A, q, r, dq, dr)
    Δq, Δr = unthunk(dq), unthunk(dr)
    dqnot0 = !(Δq isa AbstractZero)
    drnot0 = !(Δr isa AbstractZero)
    (!dqnot0 && !drnot0) && return NoTangent()
    size(r, 1) == size(r, 2) && return qr_back_fullrank(q, r, Δq, Δr)
    M, N = size(r)
    B = view(A,:,M+1:N)
    U = view(r,:,1:M)
    if drnot0
        dD = view(Δr,:,M+1:N);
        da = qr_back_fullrank(q, U, (dqnot0 ? Δq+B*dD' : B*dD'), view(Δr,:,1:M))
        db = q*dD
    else
        da = qr_back_fullrank(q, U, Δq, ZeroTangent())
        db = zero(B)
    end
    hcat(da, db)
end

"""
    lq_back(A, l, q, dl, dq) -> Matrix

backward for LQ decomposition, for an arbituary shaped input matrix.

References:
    Seeger, M., Hetzel, A., Dai, Z., Meissner, E., & Lawrence, N. D. (2018). Auto-Differentiating Linear Algebra.
    Differentiable Programming Tensor Networks, Hai-Jun Liao, Jin-Guo Liu, Lei Wang, Tao Xiang
"""
function lq_back_fullrank(L, Q, dL, dQ)
    M = ZeroAdder()
    dL === nothing || (M += L'*dL)
    dQ === nothing || (M -= dQ*Q')
    C = copyltu!(M)*Q
    if dQ !== nothing
        C += dQ
    end
    #inv(L)' * C
    trtrs!('L', 'C', 'N', L, C)
end

"""
    lq_back(A, L, Q, dL, dQ) -> Matrix

backward for QR decomposition, for an arbituary shaped input matrix.

References:
    Seeger, M., Hetzel, A., Dai, Z., Meissner, E., & Lawrence, N. D. (2018). Auto-Differentiating Linear Algebra.
    HaiJun's paper.
"""
function lq_back(A, L, Q, dL, dQ)
    ΔL, ΔQ = unthunk(dL), unthunk(dQ)
    dunot0 = !(ΔQ isa AbstractZero)
    dlnot0 = !(ΔL isa AbstractZero)
    (!dunot0 && !dlnot0) && return NoTangent()
    N, M = size(L)
    M == N && return lq_back_fullrank(L, Q, ΔL, ΔQ)
    B = view(A,M+1:N,:)
    U = view(L,1:M,:)
    if dlnot0
        dD = view(ΔL,M+1:N,:);
        da = lq_back_fullrank(U, Q, view(ΔL,1:M,:), (dunot0 ? ΔQ+dD'*B : dD'*B));
        db = dD*Q
    else
        da = lq_back_fullrank(U, Q, ZeroTangent(), ΔQ);
        db = zero(B)
    end
    vcat(da, db)
end
