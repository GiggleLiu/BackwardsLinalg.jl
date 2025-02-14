module BackwardsLinalg

using ChainRulesCore; import ChainRulesCore: rrule
using LinearAlgebra; import LinearAlgebra: ldiv!

struct ZeroAdder end
Base.:+(a, zero::ZeroAdder) = a
Base.:+(zero::ZeroAdder, a) = a
Base.:-(a, zero::ZeroAdder) = a
Base.:-(zero::ZeroAdder, a) = -a
Base.:-(zero::ZeroAdder) = zero

include("qr.jl")
include("svd.jl")
include("lstsq.jl")
include("rsvd.jl")
include("symeigen.jl")
include("chainrules.jl")

end
