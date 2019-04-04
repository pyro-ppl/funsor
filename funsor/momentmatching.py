from __future__ import absolute_import, division, print_function

from funsor.interpreter import dispatched_interpretation
from funsor.terms import eager


@dispatched_interpretation
def moment_matching(cls, *args):
    """
    A moment matching interpretation of :class:`~funsor.terms.Reduce`
    expressions. This falls back to :class:`~funsor.terms.eager` in other
    cases.
    """
    result = moment_matching.dispatch(cls, *args)
    if result is None:
        result = eager(cls, *args)
    return result


__all__ = [
    'moment_matching',
]
