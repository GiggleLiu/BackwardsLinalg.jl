module BackwardsLinalg
using LinearAlgebra
using Requires

export svd, qr, lq, symeigen, rsvd

struct ZeroAdder end
Base.:+(a, zero::ZeroAdder) = a
Base.:+(zero::ZeroAdder, a) = a
Base.:-(a, zero::ZeroAdder) = a
Base.:-(zero::ZeroAdder, a) = -a
Base.:-(zero::ZeroAdder) = zero

include("qr.jl")
include("svd.jl")
include("rsvd.jl")
include("symeigen.jl")
include("zygote.jl")

function __init__()
    @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" include("cudalib.jl")
end
end
