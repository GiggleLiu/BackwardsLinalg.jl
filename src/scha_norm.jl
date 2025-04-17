function scha_norm(A::Matrix{T}, p::Real) where T
    S = LinearAlgebra.svd(A).S
    if p == Inf
        return S[1]
    end
    return sum(S.^p)^(1/p)
end

function scha_norm_back(A,p,ā)
    a = scha_norm(A,p)
    U,S,V = LinearAlgebra.svd(A)
    if p == Inf
        Ā = ā*U[:,1]*V[:,1]'
    else
        Ā = ā*a^(1-p)*U * diagm(S.^(p-1)) *V'
    end
    return Ā
end


