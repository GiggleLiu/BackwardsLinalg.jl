# Backward functions for Linear Algebra

[![Build Status](https://travis-ci.com/GiggleLiu/BackwardsLinalg.jl.svg?branch=master)](https://travis-ci.com/GiggleLiu/BackwardsLinalg.jl)
[![Codecov](https://codecov.io/gh/GiggleLiu/BackwardsLinalg.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/GiggleLiu/BackwardsLinalg.jl)

![#f03c15](https://placehold.it/15/f03c15/000000?text=+) This project is still in progress ...

Backward functions for linear algebras,
It is currently ported to `Zygote.jl` for testing, but these porting codes will be moved to other places (like merging them to `Zygote.jl`) in the future.

## Why we need BackwardsLinalg.jl?
Not only in Julia, but also in well known machine learning packages in python like pytorch, one can hardly find a numerical stable implementations of linear algebra function. This missing piece is crutial to autodiff applications in tensor networks algorithms.

## Table of Supported Functions

Note: it will change the default behavior, we are considering not changing the output type (SVD, QR) latter when Zygote is stronger.

- [x] svd and rsvd (randomized SVD)
- [x] qr
- [ ] cholesky   # Nabla.jl
- [ ] powermethod   # we need fixed point methods, trying hard ...
- [x] eigen      # linear BP paper, only symmetric case considered
- [x] lq         # similar to qr
- [ ] pfaffian    # find it nowhere, lol

For `logdet`, `det` and `tr`, people can find it in `ChainRules.jl` and `Nabla.jl`.

Derivation of adjoint backward functions could be found [here](https://giggleliu.github.io/2019/04/02/einsumbp.html).

## How to Use
It currently ports into `Zygote.jl`
```julia
using Zygote, BackwardsLinalg

function loss(A)
    M, N = size(A)
    U, S, V = svd(A)
    psi = U[:,1]
    H = randn(ComplexF64, M, M)
    H+=H'
    real(psi'*H*psi)[]
end

a = randn(ComplexF64, 4, 6)
g = loss'(a)
```

Try something interesting (the backward of TRG code, `TensorOperations.jl` (as well as patch https://github.com/Jutho/TensorOperations.jl/pull/59) is required.)
```bash
julia test/trg.py
```
