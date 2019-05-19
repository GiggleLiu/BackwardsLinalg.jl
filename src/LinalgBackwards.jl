module LinalgBackwards
import LinearAlgebra: svd, qr, lq, eigen
using LinearAlgebra

export svd, qr, lq, eigen

struct ZeroAdder end
Base.:+(a, zero::ZeroAdder) = a
Base.:+(zero::ZeroAdder, a) = a

include("qr.jl")
include("svd_gradient.jl")
include("eigen_gradient.jl")
include("zygote.jl")

end
