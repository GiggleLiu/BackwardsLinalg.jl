#= The idea of using Cauchy integrate to calculate gradient refers to 
Torabi, Tina, Timon S. Gutleb, and Christoph Ortner. “Fast Automatically Differentiable Matrix Functions and Applications in Molecular Simulations.” arXiv, January 2, 2025. https://doi.org/10.48550/arXiv.2412.12598.

Amd the code implement of cauchy integrate refers to the reposity attached to this paper: 
https://github.com/tinatorabi/EntropyGrad.jl
=#

#= We plan to implement matrix function for f(z) who has almost one singular 0. Now I implement the entire function case 
and I will implement meromorphic function in the future.
=#

function matrix_func(f,A::AbstractMatrix{T}) where T
    @assert size(A,1) == size(A,2)
    M = opnorm(A,2)+1.0
    θ_vec = collect(0:min(0.01,1/M):0.999)*2π
    dθ = θ_vec[2] - θ_vec[1]
    sample_points = M * exp.(im*θ_vec)
    w = M * (exp.(im * (θ_vec .+ 0.5 * dθ)) - exp.(im * (θ_vec .- 0.5 * dθ)))
    res = zeros(complex(T),size(A)...)
    for i in eachindex(sample_points)
        z = sample_points[i]
        res .+= w[i] * f(z)/(2π*im) * inv(z*I - A)
    end
    if T <: Real
        res = real.(res)
    end
    return res
end

function matrix_func_back(B̄,f,A::AbstractMatrix{T}) where T
    @assert size(A,1) == size(A,2)
    M = opnorm(A,2)+1.0
    θ_vec = collect(0:min(0.01,1/M):0.999)*2π
    dθ = θ_vec[2] - θ_vec[1]
    sample_points = M * exp.(im*θ_vec)
    w = M * (exp.(im * (θ_vec .+ 0.5 * dθ)) - exp.(im * (θ_vec .- 0.5 * dθ)))
    res = zeros(complex(T),size(A)...)
    tmp = zeros(complex(T),size(A)...)
    for i in eachindex(sample_points)
        z = sample_points[i]
        tmp = inv(z*I-A)
        res .+= w[i] * f(z)/(2π*im) * tmp * B̄' * tmp
    end
    if T <: Real
        res = real.(res)
    end
    return res'
end



