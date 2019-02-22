# Funsor ![unstable](https://img.shields.io/badge/status-unstable-red.svg)

Functional analysis + tensors + symbolic algebra.

This library is an experimental work in progress.
Beware building on top of this unstable prototype.

## Design

See [design doc](https://docs.google.com/document/d/1NVlfQnNQ0Aebg8vfIGcJKsnSqAhB4bbClQrb5dwm2OM).

The goal of this library is to generalize [Pyro](http://pyro.ai)'s delayed
inference algorithms from discrete to continuous variables, and to create
machinery to enable partially delayed sampling compatible with universality. To
achieve this goal this library makes three orthogonal design choices:

1.  Functions are first class objects. Funsors generalize the tensor interface
    to also cover arbitrary functions of multiple variables ("inputs"), where
    variables may be integers, real numbers or themselves tensors. Function
    evaluation / substitution is the basic operation, generalizing tensor
    indexing.  This allows probability distributions to be first-class Funsors
    and make use of existing tensor machinery, for example we can generalize
    tensor contraction to computing analytic integrals in conjugate
    probabilistic models.

2.  Support nonstandard interpretation. Funsors support user-defined
    interpretations, including, eager, lazy, mixed eager+lazy, memoized (like
    opt\_einsum's sharing), and approximate interpretations like Monte Carlo
    approximations of integration operations (e.g. `.sum()` over a funsor
    dimension).

3.  Named dimensions. Substitution is the most basic operation of Funsors. To
    avoid the difficulties of broadcasting and advanced indexing in
    positionally-indexed tensor libraries, all Funsor dimensions are named.
    Indexing uses the `.__call__()` method and can be interpreted as
    substitution (with well-understood semantics).  Funsors are viewed as
    algebraic expressions with one algebraic free variable per dimension. Each
    dimension is either covariant (an output) or contravariant (an input).

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

- `funsor.ops` is a collection of basic ops: unary, binary, and reductions.
- `funsor.terms` contains AST classes for symbolic algebra.
- `funsor.torch` contains wrappers around PyTorch `Tensor`s and functions.
- `funsor.distributions` contains standard probability distributions.
- `funsor.interpreter` implements different evaluation strategies.
- `funsor.minipyro` a small Funsor-compatible implementation of Pyro.

## Related projects

- Pyro's [ops.packed](https://github.com/uber/pyro/blob/dev/pyro/ops/packed.py),
  [ops.einsum](https://github.com/uber/pyro/blob/dev/pyro/ops/einsum), and
  [ops.contract](https://github.com/uber/pyro/blob/dev/pyro/ops/contract.py)
- [Birch](https://birch-lang.org/)'s [delayed sampling](https://arxiv.org/abs/1708.07787)
- [autoconj](https://arxiv.org/abs/1811.11926)
- [dyna](http://www.cs.jhu.edu/~nwf/datalog20-paper.pdf)
- [PSI solver](https://psisolver.org)
- [Hakaru](https://hakaru-dev.github.io)
- [sympy](https://www.sympy.org/en/index.html)
- [namedtensor](https://github.com/harvardnlp/namedtensor)
