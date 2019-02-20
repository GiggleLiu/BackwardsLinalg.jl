using LinearAlgebra

function powermethod(A, x0::AbstractVector=randn(size(A, 2)) |> normalize!; niter::Int=1000)
    e,v = eigen(A)
    return e[1], v[:,1]
    for i = 1:niter
        x0 = A*x0
        x0 |> normalize!
    end
    x0'*A*x0, x0
end

function powermethod_back(A, e, v, de, dv)
    @show A
    de isa Nothing && (de = zero(e))
    dv isa Nothing && (dv = zero(v))
    nm = norm(A*v)
    dvAinv = dv'*inv(A)
    Δ = zero(A)
    de*v*v' + norm(A*v)*(dvAinv*v)*v*v' - dvAinv'*v'
end

@primitive powermethod(A),dy,y powermethod_back(A, y..., dy...)

xa = randn(4,4); xa+=xa'
x = Param(xa)
b = randn(4)
y = @diff powermethod(x)[2] |> sum
grad(y, x)
#e = @diff powermethod(x)[1]
#grad(e, x)

δ = 1e-5
xa_ = copy(xa)
xa_[1]+=δ/2
pos = powermethod(xa_)
xa_[1]-=δ
neg = powermethod(xa_)
(pos[1]-neg[1])/δ

using Test
@testset "power method" begin
    N = 4
    A = randn(N, N); A = A+A'
    eg, vg = powermethod(A)
    eigenres = eigen(A)
    i = argmax(eigenres.values .|> abs)
    @test eg ≈ eigenres.values[i]
    @test vg'*eigenres.vectors[:,i] |> abs ≈ 1
    @test AutoGrad.gradcheck(A->powermethod(A)[2].*b |> sum, A)
end


N = 4
A = randn(N, N); A = A+A'
eg, vg = powermethod(A)
eigenres = eigen(A)
i = argmax(eigenres.values .|> abs)
@test eg ≈ eigenres.values[i]
@test vg'*eigenres.vectors[:,i] |> abs ≈ 1
@test AutoGrad.gradcheck(A->powermethod(A)[2].*b |> sum, A)
