function lstsq(A, b)
    return A \ b
end

function lstsq_back(A, b, x, dx)
    Q, R_ = qr(A)
    R = LinearAlgebra.UpperTriangular(R_)
    y = R' \ dx
    z = R \ y
    residual = b .- A*x
    b̅ = Q * y
    return residual * z' - b̅ * x', b̅
end


