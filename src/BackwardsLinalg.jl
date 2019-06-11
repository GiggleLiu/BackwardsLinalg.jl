module BackwardsLinalg
import LinearAlgebra: svd, qr, lq, eigen
using LinearAlgebra

export svd, qr, lq, eigen, rsvd

struct ZeroAdder end
Base.:+(a, zero::ZeroAdder) = a
Base.:+(zero::ZeroAdder, a) = a
Base.:-(a, zero::ZeroAdder) = a
Base.:-(zero::ZeroAdder, a) = -a
Base.:-(zero::ZeroAdder) = zero

include("qr.jl")
include("svd.jl")
include("rsvd.jl")
include("eigen.jl")
include("zygote.jl")

end
