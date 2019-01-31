# Funsor

Functional tensors.

## Design

See [design doc](https://docs.google.com/document/d/1LUj-oV5hJe74HJWKtog07Qrcaq4uhZQ5NuYuRevaVFo).

### Goals

-   Factor out [pyro.ops.packed](https://github.com/uber/pyro/blob/dev/pyro/ops/packed.py)
    as a stand-alone library.
-   Provide named dimensions, named slicing, and name-based broadcasting.
-   Make advanced indexing well-defined via expression substitution.
-   Expose distributions as objects as funsors with real-valued dimensions.

### Non-goals

-   We provide only a mathematically minimal set of dimension types:
    bounded integer, `real`, `positive`, `unit_interval`.
    Specifically we do not provide named values.

## Related projects

- [dyna](http://www.cs.jhu.edu/~nwf/datalog20-paper.pdf) 
- [xarray](http://xarray.pydata.org/en/stable)
- [namedtensor](https://github.com/harvardnlp/namedtensor)
