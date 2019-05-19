# Backward functions for Linear Algebra

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://GiggleLiu.github.io/LinalgBackwards.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://GiggleLiu.github.io/LinalgBackwards.jl/dev)
[![Build Status](https://travis-ci.com/GiggleLiu/LinalgBackwards.jl.svg?branch=master)](https://travis-ci.com/GiggleLiu/LinalgBackwards.jl)
[![Codecov](https://codecov.io/gh/GiggleLiu/LinalgBackwards.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/GiggleLiu/LinalgBackwards.jl)

![#f03c15](https://placehold.it/15/f03c15/000000?text=+) This project is still in progress ...

Backward functions for linear algebras,
It is currently ported to `Flux.jl` for testing, but these porting codes will be moved to other places (like merging them to `Flux.jl`) in the future.

## Why we need LinearBackwards.jl?
Not only in Julia, but also in well known machine learning packages in python like pytorch, one can hardly find a numerical stable implementations of linear algebra function. This missing piece is crutial to autodiff applications in tensor networks algorithms.

## Table of Supported Functions

We put `_` before functions since we changed the rule of functions in `LinearAlgebra`, the outputs are Tuples.

- [x] _svd
- [x] qr
- [ ] _cholesky   # Nabla.jl
- [ ] _powermethod   # we need fixed point methods, trying hard ...
- [x] _eigen      # linear BP paper
- [ ] _lu         # similar to qr
- [ ] einsum      # Pytorch
- [ ] pfaffian    # find it nowhere, lol

For `logdet`, `det` and `tr`, people can find it in `ChainRules.jl` and `Nabla.jl`.

## How to Use
If you are using `Flux.jl` and want to call `svd`, please type
```julia
using Flux, LinalgAutodiff

U, S, V = _svd(A)
```
otherwise please check `svd_back` to see how it works.

Try something interesting (the backward of TRG code, `TensorOperations.jl` (as well as patch https://github.com/Jutho/TensorOperations.jl/pull/59) is required.)
```bash
julia test/trg.py
```
