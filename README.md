# Funsor

Functional analysis + tensors + term algebra.

## Design

See [design doc](https://docs.google.com/document/d/1LUj-oV5hJe74HJWKtog07Qrcaq4uhZQ5NuYuRevaVFo).
This library has two main goals.

### Goal 1: Refactor Pyro

The main goal is to replace Pyro's internal
[pyro.ops.packed](https://github.com/uber/pyro/blob/dev/pyro/ops/packed.py)
module by a stand-alone library.  This library follows patterns of first order
term algebra, and replaces positional tensor dimensions with named dimensions.
Positional dimensions are still accessible when indexing, but names are used to
determine broadcasting.

### Goal 2: First class support for probability distributions

The second goal of this library is to represent probability distributions as
tensor-like objects whose dimensions have size "real" or size "positive" etc.
This aims to allow transforms to work seamlessly on the inputs of a transformed
distribution.

The immediate application in Pyro is to generalize Pyro's discrete inference to
exact inference in all exponential family distributions.  Each multivariate
exponential family will be implemented as a `Funsor` type, e.g.
`GaussianFunsor`.

### Non-goals

This library aims at a mathematically minimal set of dimension types and
`Funsors`.  This library is intended to ease the job of inference engineers,
rather than serve as a convenient tool for data scientists.  We aim to wrap
robust implementations in a generic extensible way, but do not aim to implement
all probability distributions or tensor types.

## Related projects

- [dyna](http://www.cs.jhu.edu/~nwf/datalog20-paper.pdf) 
- [xarray](http://xarray.pydata.org/en/stable)
- [namedtensor](https://github.com/harvardnlp/namedtensor)
