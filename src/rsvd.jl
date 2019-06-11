using LinearAlgebra

"""
randomized SVD.
"""
function rsvd(A::Array{T}, k::Int=min(size(A)...), oversample::Int=10, power::Int=10, ortho::Bool=false) where T
    m, n = size(A)
    p = min(n,oversample*k)
    Y = A * randn(T, n,p)

    for i = 1:power
        Y = A * (ortho ? qr(A' * Y).Q : A' * Y)
    end
    Q, R = qr(Y)
    B = Q' * A

    U, s ,V = svd(B)
    U = Q * U
    k = min(k,size(U, 2))
    U = U[:,1:k]
    V = V[:,1:k]
    s = s[1:k]
    return U,s,V
end
