from collections import OrderedDict
from contextlib import contextmanager

from funsor.integrate import Integrate
from funsor.interpreter import dispatched_interpretation, interpretation
from funsor.terms import Funsor, eager


@dispatched_interpretation
def monte_carlo(cls, *args):
    """
    A Monte Carlo interpretation of :class:`~funsor.integrate.Integrate`
    expressions. This falls back to :class:`~funsor.terms.eager` in other
    cases.
    """
    # TODO Memoize sample statements in a context manager.
    result = monte_carlo.dispatch(cls, *args)(*args)
    if result is None:
        result = eager(cls, *args)
    return result


# This is a globally configurable parameter to draw multiple samples.
monte_carlo.sample_inputs = OrderedDict()


@contextmanager
def monte_carlo_interpretation(**sample_inputs):
    """
    Context manager to set ``monte_carlo.sample_inputs`` and
    install the :func:`monte_carlo` interpretation.
    """
    old = monte_carlo.sample_inputs
    monte_carlo.sample_inputs = OrderedDict(sample_inputs)
    try:
        with interpretation(monte_carlo):
            yield
    finally:
        monte_carlo.sample_inputs = old


@monte_carlo.register(Integrate, Funsor, Funsor, frozenset)
def monte_carlo_integrate(log_measure, integrand, reduced_vars):
    sample = log_measure.sample(reduced_vars, monte_carlo.sample_inputs)
    if sample is log_measure:
        return None  # cannot progress
    reduced_vars |= frozenset(monte_carlo.sample_inputs).intersection(sample.inputs)
    return Integrate(sample, integrand, reduced_vars)


__all__ = [
    'monte_carlo',
    'monte_carlo_interpretation'
]
