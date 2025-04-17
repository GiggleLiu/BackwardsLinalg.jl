function arg_lstsq(A::Matrix{T},b::Vector{T}) where T<:Number
    A1=A'*A
    @assert LinearAlgebra.det(A1)!=0
    return A1\(A'*b)
end

function arg_lstsq_back(A::Matrix{T},b::Vector{T},x,x̄) where T
    Q,R = LinearAlgebra.qr(A)
    b̄ = Q*(R')^(-1)*x̄
    Ā = (b-A*x)*x̄'*(R'*R)^(-1) -Q*(R')^(-1)*x̄*x'
    return Ā,b̄
end

function lstsq(A::Matrix{T}, b::Vector{T}) where T
    U,_,_ = LinearAlgebra.svd(A)
    return real(b'*(LinearAlgebra.I-U*U')*b)
end

function lstsq_back(A::Matrix{T}, b::Vector{T} ,ā) where T
    U,S,V = LinearAlgebra.svd(A)
    U = Matrix(U)
    V = Matrix(V)
    b̄ = 2 * ā * (LinearAlgebra.I - U*U') * b
    Ū = -2 * ā * b * b' * U
    S̄ = zero(S)
    V̄ = zero(V)
    Ā = svd_back(U,S,V,Ū,S̄,V̄)
    return Ā, b̄
end


