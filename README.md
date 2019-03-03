# Backward functions for Linear Algebra

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://GiggleLiu.github.io/LinalgBackwards.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://GiggleLiu.github.io/LinalgBackwards.jl/dev)
[![Build Status](https://travis-ci.com/GiggleLiu/LinalgBackwards.jl.svg?branch=master)](https://travis-ci.com/GiggleLiu/LinalgBackwards.jl)
[![Codecov](https://codecov.io/gh/GiggleLiu/LinalgBackwards.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/GiggleLiu/LinalgBackwards.jl)

Backward functions for linear algebras,
It is currently ported to `Flux.jl` for testing, but these porting codes will be moved to other places (like `Flux.jl`) in the future.

## Table of Supported Functions

We put `_` before functions since we changed the rule of functions in `LinearAlgebra`, the outputs are Tuples.

- [*] _svd
- [*] _qr
- [ ] _cholesky   # in Nabla.jl, there is an implementation
- [ ] _powermethod
- [ ] _eigen
- [ ] _inv
- [ ] _lu

## How to Use
If you are using `Flux.jl` and want to call `svd`, please type
```julia
using Flux, LinalgAutodiff

U, S, V = _svd(A)
```
otherwise please check `svd_back` to see how it works.

Try something interesting (the backward of TRG code, `TensorOperations.jl` is required.)
```bash
julia test/trg.py
```
