function sdp(C::Matrix{T}, A::Vector{Matrix{T}}, b::Vector{T}) where T
    n = size(C, 1)
    model = Model(SCS.Optimizer)
    @variable(model, X[1:n, 1:n], PSD)
    @objective(model, Min, tr(C * X))
    m = length(A)
    for i in 1:m
        @constraint(model, tr(A[i] * X) == b[i])
    end
    set_silent(model)
    optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
        return value.(X)
    end
end

function sdp_back(C::Matrix{T}, A::Vector{Matrix{T}}, b::Vector{T}, X::Matrix{T}, X̄::Matrix{T}) where T
    X = (X + X') / 2
    X̄ = (X̄ + X̄') / 2
    m = length(A)
    n = size(X, 1)
    E,U = LinearAlgebra.eigen(X)
    U = Matrix(U)
    idx = findall(E .> 1e-3)
    U = U[:,idx]
    E = E[idx]
    k = length(E)
    B = zeros(T, m, k)
    for i in 1:m
        B[i,:] = LinearAlgebra.diag(U'*A[i]*U)
    end
    S̄ = (U'*X̄*U) .* Matrix(LinearAlgebra.I(k))
    S̄ = LinearAlgebra.diag(S̄)
    B̄,b̄ = arg_lstsq_back(B,b,E,S̄) 
    Ā = Vector{Matrix{T}}(undef, m)
    for i in 1:m
        Ā[i] = U * LinearAlgebra.diagm(B̄[i,:]) * U'
    end

    return zero(C), Ā, b̄
end