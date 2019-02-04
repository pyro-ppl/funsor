# Funsor ![unstable](https://img.shields.io/badge/status-unstable-red.svg)

Functional analysis + tensors + symbolic algebra.

This library is an experimental work in progress.
Beware building on top of this unstable prototype.

## Design

See [design doc](https://docs.google.com/document/d/1LUj-oV5hJe74HJWKtog07Qrcaq4uhZQ5NuYuRevaVFo).

The goal of this library is to generalize [Pyro](http://pyro.ai)'s delayed
inference algorithms from discrete to continuous variables, and to create
machinery to enable partially delayed sampling compatible with universality. To
achieve this goal this library makes three independent design choices:

1.  Combine eager + lazy evaluation. `Funsor`s can be either eager
    (`funsor.Tensor` simply wraps `torch.Tensor`) or lazy. Lazy funsors are
    expressions that can involve eager `Tensor`s. This aims to allow partially
    delayed sampling and partial inference during Pyro program execution.

2.  Allow real-valued tensor dimensions. `Funsor` generalizes the tensor
    interface to also cover arbitrary functions of multiple variables. Indexing
    is interpreted as function evaluation. This allows probability
    distributions to be first-class `Funsor`s and make use of existing tensor
    machinery, for example we can generalize tensor contraction to computing
    analytic integrals in conjugate probilistic models.

3.  Named dimensions. To avoid the difficulties of broadcasting and advanced
    indexing, all `Funsor` dimensions are named, and indexing uses the
    `__call__` operator. `Funsors` are viewed as quantities with one algebraic
    free variable per dimension.

Using `funsor` we can easily implement Pyro-style
[delayed sampling](http://pyro.ai/examples/enumeration.html), roughly:

```py
trace_log_prob = 0.

def pyro_sample(name, dist, obs=None):
    assert isinstance(dist, Funsor)
    if obs is not None:
        value = obs
    elif lazy:
        # delayed sampling (like Pyro's parallel enumeration)
        value = funsor.Variable(name, dist.support)
    else:
        value = dist.sample('value')[0]['value']

    # save log_prob in trace
    trace_log_prob += dist(value)

    return value

# ...later during inference...
log_prob = trace_log_prob.logsumexp()  # collapses delayed variables
loss = -funsor.eval(log_prob)          # performs variable elimination
```
See [examples/minipyro.py](examples/minipyro.py) for a more complete example.

## Code organization

- `funsor.ops` is a collection of basic ops: unary, binary, and reductions
- `funsor.term` contains AST classes for symbolic algebra, including concrete
  PyTorch `Tensor`s
- `funsor.engine` contains algorithms for symbolic computation including
  variable elimination. Its entire interface is `funsor.eval()`.
- `funsor.distributions` contains standard probability distributions.

## Related projects

- Pyro's [ops.packed](https://github.com/uber/pyro/blob/dev/pyro/ops/packed.py),
  [ops.einsum](https://github.com/uber/pyro/blob/dev/pyro/ops/einsum), and
  [ops.contract](https://github.com/uber/pyro/blob/dev/pyro/ops/contract.py)
- [dyna](http://www.cs.jhu.edu/~nwf/datalog20-paper.pdf)
- [PSI solver](https://psisolver.org)
- [Hakaru](https://hakaru-dev.github.io)
- [sympy](https://www.sympy.org/en/index.html)
- [namedtensor](https://github.com/harvardnlp/namedtensor)
