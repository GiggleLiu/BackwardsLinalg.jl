export svd, svd!, svd_back
using LinearAlgebra: I, Diagonal

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

    J = F.*(transpose(U)*dU)
    K = F.*(transpose(V)*dV)

    Su = (J+J')*Diagonal(S)
    Sv = Diagonal(S) * (K+K')

    conj(U) * (Su + Sv + Diagonal(dS)) * transpose(V) +
    transpose(V * Diagonal(Sinv) * transpose(dU) * (I - U*U')) +
    conj(U * Diagonal(Sinv) * transpose(dV) * (I - V*V'))
end

function svd!(A)
    U, S, V = LinearAlgebra.svd!(A)
    U, S, Matrix(V)
end

svd(A) = svd!(A |> copy)
