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
and generalize tensor contraction to analytic integrals. This will allow us to implement a less hacky version of Pyro's parallel enumeration, roughly

```py
def pyro_sample(name, dist):
    if lazy:  # i.e. if infer["enumerate"] == "parallel"
        value = funsor.var(name, dist.support)
    else:
        value, log_prob = dist.sample(value)
        # discard log_prob since it should be normalized
    # save dist in trace, so later we can compute
    #   log_prob = dist(value=value)
    return value
```

## To Do

- implement `funsor.normal` etc. as single `Funsor`s
- fork minipyro.py and add pyro examples
- sketch optimizer for dyna/opt\_einsum
- switch from `(dims, shape)` to `schema` `OrderedDict`
- make a Bach chorale example with real + discrete latent variables

## Related projects

- Pyro's [ops.packed](https://github.com/uber/pyro/blob/dev/pyro/ops/packed.py),
  [ops.einsum](https://github.com/uber/pyro/blob/dev/pyro/ops/einsum), and
  [ops.contract](https://github.com/uber/pyro/blob/dev/pyro/ops/contract.py)
- [dyna](http://www.cs.jhu.edu/~nwf/datalog20-paper.pdf) 
- [sympy](https://www.sympy.org/en/index.html)
- [namedtensor](https://github.com/harvardnlp/namedtensor)
