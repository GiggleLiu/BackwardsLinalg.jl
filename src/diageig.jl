import LinearAlgebra: eigvals, eigvecs, eigen, Diagonal
using LinearAlgebra

export _diageig, diageig_back, sym

function _diageig(A)
    sA = Symmetric(A)
    eigval, eigvec = eigvals(sA), eigvecs(sA)
    eigval, eigvec'
end

function sym(A)
    (A + A') / 2.
end

function h(t; ϵ=0.00000001)
    max(abs(t), ϵ)*sign(t)
end

function fij(λ)
    nλ = length(λ)
    λmat = kron(λ,ones(nλ)')
    dλmat = λmat-λmat' + 0.000001 * LinearAlgebra.I
    (ones(nλ,nλ)-LinearAlgebra.I) ./ h.(dλmat)
end

function diageig_back(λ, U, dλ, dU)
    U' * (sym((dU * U') .* fij(λ)) + Diagonal(dλ)) * U
end

