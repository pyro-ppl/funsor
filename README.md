# Funsor

Functional analysis + tensors + algebra.

## Design

See [design doc](https://docs.google.com/document/d/1LUj-oV5hJe74HJWKtog07Qrcaq4uhZQ5NuYuRevaVFo).

The goal of this library is to generalize Pyro's tensor variable elimination
algorithm from discrete to continuous variables, enabling delayed sampling in
Pyro.  This library is intended as a replacement for PyTorch inside Pyro and
possibly in user-facing Pyro code (where in particular this library cleans up
advanced indexing).

The core idea is to generalize tensor dimensions to algebraic free variables
and generalize tensor contraction to analytic integrals.

## Related projects

- Pyro's [ops.packed](https://github.com/uber/pyro/blob/dev/pyro/ops/packed.py),
  [ops.einsum](https://github.com/uber/pyro/blob/dev/pyro/ops/einsum), and
  [ops.contract](https://github.com/uber/pyro/blob/dev/pyro/ops/contract.py)
- [dyna](http://www.cs.jhu.edu/~nwf/datalog20-paper.pdf) 
- [sympy](https://www.sympy.org/en/index.html)
- [xarray](http://xarray.pydata.org/en/stable)
- [namedtensor](https://github.com/harvardnlp/namedtensor)
