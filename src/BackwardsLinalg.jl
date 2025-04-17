module BackwardsLinalg

using ChainRulesCore; import ChainRulesCore: rrule
using LinearAlgebra; import LinearAlgebra: ldiv!
using JuMP, GLPK, Zygote, SkewLinearAlgebra, SCS, FFTW, NFFT, NFFTTools

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
include("norm_anlfunc.jl")
include("cls.jl")
include("det.jl")
include("inv.jl")
include("lneq.jl")
include("lp.jl")
include("sdp.jl")
include("lu.jl")
include("mxmul.jl")
include("scha_norm.jl")
include("gmres.jl")
include("pf.jl")
include("normeigen.jl")
include("fft.jl")
include("unfft.jl")
include("matrix_func.jl")

include("chainrules.jl")


end
