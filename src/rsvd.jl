# using LinearAlgebra

"""
randomized SVD.
"""
function rsvd(A::AbstractArray{T}, k::Int=min(size(A)...), oversample::Int=10, power::Int=10, ortho::Bool=false) where T
    m, n = size(A)
    p = min(n,oversample*k)
    Y = A * randn(T, n,p)

    for i = 1:power
        Y = A * (ortho ? LinearAlgebra.qr(A' * Y).Q : A' * Y)
    end
    Q, R = LinearAlgebra.qr(Y)
    B = Q' * A

    U, s ,V = LinearAlgebra.svd(B)
    U = Q * U
    k = min(k,size(U, 2))
    U = U[:,1:k]
    V = V[:,1:k]
    s = s[1:k]
    return U,s,V
end
