# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
from collections import OrderedDict

from funsor.cnf import Contraction
from funsor.delta import Delta
from funsor.gaussian import Gaussian
from funsor.integrate import Integrate
from funsor.interpretations import StatefulInterpretation
from funsor.tensor import Tensor
from funsor.terms import Approximate, Funsor, Number, Subs, Unary
from funsor.util import get_backend

from . import ops


class MonteCarlo(StatefulInterpretation):
    """
    A Monte Carlo interpretation of :class:`~funsor.integrate.Integrate`
    expressions. This falls back to the previous interpreter in other cases.

    :param rng_key:
    """

    def __init__(self, *, rng_key=None, **sample_inputs):
        super().__init__("monte_carlo")
        self.rng_key = rng_key
        self.sample_inputs = OrderedDict(sample_inputs)


@MonteCarlo.register(Integrate, Funsor, Funsor, frozenset)
def monte_carlo_integrate(state, log_measure, integrand, reduced_vars):
    sample_options = {}
    if state.rng_key is not None and get_backend() == "jax":
        import jax

        sample_options["rng_key"], state.rng_key = jax.random.split(state.rng_key)

    sample = log_measure.sample(reduced_vars, state.sample_inputs, **sample_options)
    if sample is log_measure:
        return None  # cannot progress

    return Integrate(sample, integrand, reduced_vars)


@MonteCarlo.register(Approximate, ops.LogaddexpOp, Funsor, Funsor, frozenset)
def monte_carlo_approximate(state, op, model, guide, approx_vars):
    sample_options = {}
    if state.rng_key is not None and get_backend() == "jax":
        import jax

        sample_options["rng_key"], state.rng_key = jax.random.split(state.rng_key)

    sample = guide.sample(approx_vars, state.sample_inputs, **sample_options)
    if sample is guide:
        return model  # cannot progress
    result = sample + model - guide

    return result


@functools.singledispatch
def extract_samples(discrete_density):
    """
    Extract sample values out of a funsor Delta, possibly scaled by Tensors.
    This is useful for extracting sample tensors from a Monte Carlo
    computation.
    """
    raise ValueError(
        f"Could not extract support from {type(discrete_density).__name__}"
    )


@extract_samples.register(Delta)
def _extract_samples_delta(discrete_density):
    return {name: point for name, (point, log_density) in discrete_density.terms}


@extract_samples.register(Contraction)
def _extract_samples_contraction(discrete_density):
    assert not discrete_density.reduced_vars
    result = {}
    for term in discrete_density.terms:
        result.update(extract_samples(term))
    return result


@extract_samples.register(Subs)
@extract_samples.register(Number)
@extract_samples.register(Tensor)
@extract_samples.register(Gaussian)
@extract_samples.register(Unary)
def _extract_samples_scale(discrete_density):
    return {}


__all__ = [
    "MonteCarlo",
]
