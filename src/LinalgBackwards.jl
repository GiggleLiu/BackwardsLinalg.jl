module LinalgBackwards
import LinearAlgebra: svd, qr, lq

export svd, qr, lq

struct ZeroAdder end
Base.:+(a, zero::ZeroAdder) = a
Base.:+(zero::ZeroAdder, a) = a

include("qr.jl")
include("svd_gradient.jl")
include("zygote.jl")

end
