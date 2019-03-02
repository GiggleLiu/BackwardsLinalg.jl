# Backward functions for Linear Algebra

Backward functions for linear algebras,
It is currently ported to `Flux.jl` for testing, but these porting codes will be moved to other places (like `Flux.jl`) in the future.

## Table of Supported Functions

We put `_` before functions since we changed the rule of functions in `LinearAlgebra`, the outputs are Tuples.

- [x] _svd
- [x] _qr
- [ ] _cholesky   # in Nabla.jl, there is already an implementation
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

Try something interesting (the backward of TRG code, `TensorOperations.jl` (as well as patch https://github.com/Jutho/TensorOperations.jl/pull/59) is required.)
```bash
julia test/trg.py
```
