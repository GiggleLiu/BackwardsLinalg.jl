export svd, svd!, svd_back

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

    UdU = U'*dU
    VdV = V'*dV

    Su = (F.*(UdU-UdU'))*LinearAlgebra.Diagonal(S)
    Sv = LinearAlgebra.Diagonal(S) * (F.*(VdV-VdV'))

    U * (Su + Sv + LinearAlgebra.Diagonal(dS)) * V' +
    (LinearAlgebra.I - U*U') * dU*LinearAlgebra.Diagonal(Sinv) * V' +
    U*LinearAlgebra.Diagonal(Sinv) * dV' * (LinearAlgebra.I - V*V')
end

function svd!(A)
    U, S, V = LinearAlgebra.svd!(A)
    U, S, Matrix(V)
end

svd(A) = svd!(A |> copy)
