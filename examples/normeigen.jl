using BackwardsLinalg

A = rand(ComplexF64,3,3)

A += A'

BackwardsLinalg.normeigen(A)[1]