# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

from funsor.integrate import Integrate
from funsor.interpreter import StatefulInterpretation
from funsor.terms import Funsor


class MonteCarlo(StatefulInterpretation):
    """
    A Monte Carlo interpretation of :class:`~funsor.integrate.Integrate`
    expressions. This falls back to the previous interpreter in other cases.

    :param rng_key:
    """
    def __init__(self, *, rng_key=None, **sample_inputs):
        self.rng_key = rng_key
        self.sample_inputs = OrderedDict(sample_inputs)


@MonteCarlo.register(Integrate, Funsor, Funsor, frozenset)
def monte_carlo_integrate(state, log_measure, integrand, reduced_vars):
    # FIXME use state.rng_key to here
    sample = log_measure.sample(reduced_vars, state.sample_inputs)
    if sample is log_measure:
        return None  # cannot progress
    reduced_vars |= frozenset(state.sample_inputs).intersection(sample.inputs)
    return Integrate(sample, integrand, reduced_vars)


__all__ = [
    'MonteCarlo',
]
