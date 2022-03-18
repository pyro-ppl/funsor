[![Build Status](https://github.com/pyro-ppl/funsor/workflows/CI/badge.svg)](https://github.com/pyro-ppl/funsor/actions)
[![Latest Version](https://badge.fury.io/py/funsor.svg)](https://pypi.python.org/pypi/funsor)
[![Documentation Status](https://readthedocs.org/projects/funsor/badge)](http://funsor.readthedocs.io)

# Funsor

Funsor is a tensor-like library for functions and distributions.

See
[Functional tensors for probabilistic programming](https://arxiv.org/abs/1910.10775)
for a system description.

## Installing

**Install using pip:**

Funsor supports Python 3.7+.

```sh
pip install funsor
```

**Install from source:**
```sh
git clone git@github.com:pyro-ppl/funsor.git
cd funsor
git checkout master
pip install .
```

## Using funsor

Funsor can be used through a number of interfaces:

-   Funsors can be used directly for probabilistic computations, using PyTorch
    optimizers in a standard training loop. Start with these examples:
    [discrete_hmm](examples/discrete_hmm.py),
    [eeg_slds](examples/eeg_slds.py),
    [kalman_filter](examples/kalman_filter.py),
    [pcfg](examples/pcfg.py),
    [sensor](examples/sensor.py),
    [slds](examples/slds.py), and
    [vae](examples/vae.py).
-   Funsors can be used to implement custom inference algorithms within Pyro,
    using custom elbo implementations in standard
    [pyro.infer.SVI](http://docs.pyro.ai/en/stable/inference_algos.html#pyro.infer.svi.SVI)
    training. See these examples:
    [mixed_hmm](examples/mixed_hmm/model.py) and
    [bart forecasting](https://github.com/pyro-ppl/sandbox/blob/master/2019-08-time-series/bart/forecast.py).
-   [funsor.pyro](https://funsor.readthedocs.io/en/latest/pyro.html) provides a
    number of Pyro-compatible (and PyTorch-compatible) distribution classes
    that use funsors under the hood, as well
    [utilities](https://funsor.readthedocs.io/en/latest/pyro.html#module-funsor.pyro.convert)
    to convert between funsors and distributions.
-   [funsor.minipyro](https://funsor.readthedocs.io/en/latest/minipyro.html)
    provides a limited alternate backend for the Pyro probabilistic programming
    language, and can perform some ELBO computations exactly.

## Design

See [design doc](https://docs.google.com/document/d/1NVlfQnNQ0Aebg8vfIGcJKsnSqAhB4bbClQrb5dwm2OM). 

The goal of this library is to generalize [Pyro](http://pyro.ai)'s delayed
inference algorithms from discrete to continuous variables, and to create
machinery to enable partially delayed sampling compatible with universality. To
achieve this goal this library makes three orthogonal design choices:

1.  Open terms are objects. Funsors generalize the tensor interface
    to also cover arbitrary functions of multiple variables ("inputs"), where
    variables may be integers, real numbers, or real tensors. Function
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
loss = -trace_log_prob.reduce(logaddexp)  # collapses delayed variables
```
See [funsor/minipyro.py](funsor/minipyro.py) for complete implementation.

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

## Citation

If you use Funsor, please consider citing:
```
@article{obermeyer2019functional,
  author = {Obermeyer, Fritz and Bingham, Eli and Jankowiak, Martin and
            Phan, Du and Chen, Jonathan P},
  title = {{Functional Tensors for Probabilistic Programming}},
  journal = {arXiv preprint arXiv:1910.10775},
  year = {2019}
}
```
