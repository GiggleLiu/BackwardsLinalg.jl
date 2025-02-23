function rrule(::typeof(qr), A)
	Q, R = qr(A)
    function pullback(dy)
        ΔA = @thunk qr_back(A, Q, R, unthunk.(dy)...)
        return (NoTangent(), ΔA)
    end
    return (Q, R), pullback
end

function rrule(::typeof(qr), A::AbstractMatrix, pivot::Val{true})
	Q, R, P = qr(A, pivot) 
    function pullback(dy)
        ΔA = @thunk qr_back(Q*R, Q, R, unthunk(dy[1]), unthunk(dy[2]))*P'
        return (NoTangent(), ΔA, NoTangent())
    end
    return (Q, R, P), pullback
end

function rrule(::typeof(lq), A)
    L, Q = lq(A)
    function pullback(dy)
        ΔA = @thunk lq_back(A, L, Q, unthunk.(dy)...)
        return (NoTangent(), ΔA)
    end
    return (L, Q), pullback
end

function rrule(::typeof(svd), A)
    U, S, V = svd(A)
    @info "svd forward" U S V
    function pullback(dy)
        @info "svd pullback"
        ΔA = @thunk svd_back(U, S, V, unthunk.(dy)...)
        return (NoTangent(), ΔA)
    end
    return (U, S, V), pullback
end

function rrule(::typeof(rsvd), A, args...; kwargs...)
    U, S, V = rsvd(A, args...; kwargs...)
    function pullback(dy)
        ΔA = @thunk svd_back(U, S, V, unthunk.(dy)...)
        return (NoTangent(), ΔA)
    end
    return (U, S, V), pullback
end

function rrule(::typeof(symeigen), A)
    E, U = symeigen(A)
    function pullback(dy)
        ΔA = @thunk symeigen_back(E, U, unthunk.(dy)...)
        return (NoTangent(), ΔA)
    end
    return (E, U), pullback
end

function rrule(::typeof(lstsq), A, b)
	x = lstsq(A, b) 
    function pullback(dy)
        Δy = unthunk(dy)
        ΔA, Δb = @thunk lstsq_back(A, b, x, Δy)
        return (NoTangent(), ΔA, Δb)
    end
    return x, pullback
end