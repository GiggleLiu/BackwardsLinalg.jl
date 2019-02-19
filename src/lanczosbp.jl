using LinearAlgebra
using IterativeSolvers
import LinearAlgebra: ldiv!

struct Projector{VT<:AbstractVector}
    v::VT
end

ldiv!(p::Projector, x) = x - p.v*(p.v'*x)
ldiv!(x, p::Projector) = x - p.v'*(x*p.v)


gs(A::AbstractMatrix) = (ei = eigen(A); (ei.values[1], ei.vectors[:,1]))
function gs_backward(A, e, v, de, dv)
    #x = gmres((I-v*v')*(e*I-A), (I-v*v')*dv)
    x = gmres((e*I-A)*(I-v*v'), dv)'
    #x = gmres(e*I-A, dv, Pl=Projector(v))
    #x = gmres(e*I-A, (I-v*v')*dv, Pl=Projector(v))
    #x = gmres(e*I-A, (I-v*v')*dv)
    #(de*v + (I-v*v')*pinv(e*I-A)*dv)*v'
    (de*v + x)*v'
end

function num_diff(A, de, dv)
    δ = 1e-5
    A = A |> copy
    dA = zeros(n, n)
    for i = 1:n
        for j = 1:n
            A[i,j]+=δ/2
            pos = gs(A)
            A[i,j]-=δ
            neg = gs(A)
            A[i,j]+=δ/2
            grad = (pos.-neg)./δ
            dA[i,j] = grad[1]*de + sum(grad[2].*dv)
        end
    end
    dA
end

n=3
A = randn(n,n); A += A'
e,v = gs(A)
de = randn()
dv = randn(n)
de = 0
dv = [1.0, 0,0]
dA = gs_backward(A, e, v, de, dv)


@show num_diff(A, e, v, de, dv)
@show dA

using Test
@test isapprox(dA .|> abs, num_diff() .|> abs)
