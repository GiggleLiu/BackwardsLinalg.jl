function lp(c::Vector{T},A::Matrix{T}, b::Vector{T}) where T<:Real
    n = size(A,2)
    model = JuMP.Model(GLPK.Optimizer)
    @variable(model, x[1:n] >= 0); 
    @objective(model, Min, dot(c, x))
    @constraint(model, A * x .== b); 
    JuMP.optimize!(model)
    return JuMP.value.(x),JuMP.objective_value(model)
end

function lp_back(c::Vector{T},A::Matrix{T}, b::Vector{T}, x::Vector, x̄0, ā) where T<:Real
    x̄0 = (x̄0 === nothing ? zero(x) : x̄0)
    ā = (ā === nothing ? T(0) : ā)


    x̄ = x̄0 + ā*c
    c̄ = ā * x
    bsc_vrb = findall(x -> abs(x)>1e-12,x)

    xB = copy(x[bsc_vrb])
    B = copy(A[:,bsc_vrb])
    x̄B = x̄[bsc_vrb]
    B̄, b̄ = lneq_back(B,b,xB,x̄B)
    Ā = zero(A)
    Ā[:,bsc_vrb] = copy(B̄)
    return c̄, Ā, b̄
end