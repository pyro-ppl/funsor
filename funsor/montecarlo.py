from __future__ import absolute_import, division, print_function

from collections import OrderedDict

from contextlib2 import contextmanager

from funsor.integrate import Integrate, integrator
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
    result = monte_carlo.dispatch(cls, *args)
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
@integrator
def monte_carlo_integrate(log_measure, integrand, reduced_vars):
    sample = log_measure.sample(reduced_vars, monte_carlo.sample_inputs)
    if sample is log_measure:
        return None  # cannot progress
    reduced_vars |= frozenset(monte_carlo.sample_inputs).intersection(sample.inputs)
    return Integrate(sample, integrand, reduced_vars)


def resample(log_measure, log_weights, resampled_vars):
    probs = (log_weights - log_weights.reduce(ops.logaddexp, resampled_vars)).exp()
    indices_dist = dist.Categorical(probs=probs)
    fresh_name_subs = get_fresh_names(resampled_vars)
    indices_dist = indices_dist(**fresh_name_subs)
    sample_inputs = OrderedDict((k, log_measure.inputs[k]) for k in resampled_vars)
    indices = indices_dist.unscaled_sample(frozenset(fresh_name_subs), sample_inputs=sample_inputs)


@dispatched_interpretation
def sequential_monte_carlo(cls, *args):
    """
    A moment matching interpretation of :class:`Reduce` expressions. This falls
    back to :class:`eager` in other cases.
    """
    result = sequential_monte_carlo.dispatch(cls, *args)
    if result is None:
        result = eager(cls, *args)
    return result

sequential_monte_carlo.sample_inputs = OrderedDict()


@sequential_monte_carlo.register(Integrate, Funsor, Funsor, frozenset)
@integrator
def sequential_monte_carlo_integrate(log_measure, integrand, reduced_vars):
    resampled_vars = frozenset(sequential_monte_carlo.sample_inputs).intersection(integrand.inputs)
    log_weights = Integrate(log_measure, integrand, frozenset(log_measure.inputs) - resampled_vars)
    # TODO don't double-count weights
    if isinstance(log_weights, Tensor):
        log_measure = resample(log_measure, log_weights, resampled_vars)
    else:
        warnings.warn("Could not resample vars {} with log_weights of type {}".format(resampled_vars, type(log_weights).__name__))

    sample = log_measure.sample(reduced_vars, sequential_monte_carlo.sample_inputs)
    if sample is log_measure:
        return None  # cannot progress

    reduced_vars |= frozenset(sequential_monte_carlo.sample_inputs).intersection(sample.inputs)
    return Integrate(sample, integrand, reduced_vars)


__all__ = [
    'monte_carlo',
    'monte_carlo_interpretation',
    'sequential_monte_carlo',
    'sequential_monte_carlo_interpretation',
]
