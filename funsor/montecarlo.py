# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

from funsor.integrate import Integrate
from funsor.interpretations import StatefulInterpretation
from funsor.terms import Approximate, Funsor
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
    if state.sample_inputs:
        sample -= math.log(sum(v.size for v in state.sample_inputs.values))
    if sample is log_measure:
        return None  # cannot progress

    # FIXME should we also avoid reducing over sample_inputs?
    reduced_vars |= frozenset(
        v for v in sample.input_vars if v.name in state.sample_inputs
    )

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


__all__ = [
    "MonteCarlo",
]
