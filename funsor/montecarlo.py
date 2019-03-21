from __future__ import absolute_import, division, print_function

from collections import OrderedDict

from contextlib2 import contextmanager

from funsor.interpreter import dispatched_interpretation, interpretation
from funsor.terms import eager


@dispatched_interpretation
def monte_carlo(cls, *args):
    # TODO Memoize sample statements in a context manager.
    result = monte_carlo.dispatch(cls, *args)
    if result is None:
        result = eager(cls, *args)
    return result


# This is a globally configurable parameter to draw multiple samples.
monte_carlo.sample_inputs = OrderedDict()


@contextmanager
def monte_carlo_interpretation(**sample_inputs):
    old = monte_carlo.sample_inputs
    monte_carlo.sample_inputs = OrderedDict(sample_inputs)
    try:
        with interpretation(monte_carlo):
            yield
    finally:
        monte_carlo.sample_inputs = old


__all__ = [
    'monte_carlo',
    'monte_carlo_interpretation'
]
