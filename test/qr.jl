using BackwardsLinalg
using Test, Random, Zygote

function gradient_check(f, args...; η = 1e-5)
    g = Zygote.gradient(f, args...)
    dy_expect = η*sum(abs2.(g[1]))
    dy = f(args...)-f([gi === nothing ? arg : arg.-η.*gi for (arg, gi) in zip(args, g)]...)
    @show dy
    @show dy_expect
    isapprox(dy, dy_expect, rtol=1e-2, atol=1e-8)
end

@testset "qr Q complex" begin
    T = ComplexF64
    Random.seed!(3)
    for (M, N) in [(2, 6), (4, 4), (6, 2)]
        @show M, N
        A = randn(T, M, N)
        op = randn(M, M)
        op += op'
        op2 = randn(N, N)
        op2 += op2'

        function tfunc(x)
            Q, R = BackwardsLinalg.qr(x)
            v = Q[:,1]
            v2 = R[2,:]
            (v'*op*v + v2'*op2*v2)[] |> real
        end
        @test gradient_check(tfunc, A)
    end
    a = [1+1im 2+1im; 3-1im 4+2im]
    @test BackwardsLinalg.copyltu!(a) ≈ [1 3+1im; 3-1im 4]
end

@testset "lq" begin
    Random.seed!(3)
    T = ComplexF64
    for (M, N) in [(2, 6), (4, 4), (6, 2)]
        @show M, N
        A = randn(T, M, N)

        op = randn(M, M)
        op += op'
        op2 = randn(N, N)
        op2 += op2'

        function tfunc(x)
            L, Q = BackwardsLinalg.lq(x)
            v = L[:,1]
            v2 = Q[2,:]
            (v'*op*v + v2'*op2*v2)[] |> real
        end
        @test gradient_check(tfunc, A)
    end
end

@testset "qr Q complex pivot" begin
    T = Float64
    Random.seed!(3)
    for (M, N) in [(20, 60), (40, 40), (60, 20)]
        A = randn(T, M, N)
        op = randn(M, M)
        op += op'
        op2 = randn(N, N)
        op2 += op2'

        function tfunc(x)
            Q, R, P = BackwardsLinalg.qr(x, Val(true))
            v = Q[:,1]
            v2 = R[2,:]
            (v'*op*v + v2'*op2*v2)[] |> real
        end
        @show tfunc(A)
        @test gradient_check(tfunc, A)
    end
    a = [1+1im 2+1im; 3-1im 4+2im]
    @test BackwardsLinalg.copyltu!(a) ≈ [1 3+1im; 3-1im 4]
end
