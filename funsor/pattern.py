from __future__ import absolute_import, division, print_function

from collections import defaultdict

import funsor.ops as ops
from funsor.terms import Binary, Number, Reduce, Unary
from funsor.torch import Tensor


def try_match_reduction(op, x):
    if isinstance(x, Reduce) and x.op is op:
        yield x.arg, x.reduced_vars


def try_match_tensors(operands):
    if all(isinstance(x, Tensor) for x in operands):
        yield operands


def match_commutative(op, *args):
    assert callable(op)
    pending = list(args)
    terms = []
    while pending:
        x = pending.pop()
        if isinstance(x, Binary):
            if x.op is op:
                pending.append(x.lhs)
                pending.append(x.rhs)
            elif op is ops.add and x.op is ops.sub:
                pending.append(x.lhs)
                pending.append(-x.rhs)
            else:
                terms.append(x)
        else:
            terms.append(x)
    return terms


def simplify_sum(x):
    """
    This attempts to cancel terms like ``x + y - x = y``.
    """
    counts = defaultdict(int)
    for term in match_commutative(ops.add, x):
        sign = 1
        while isinstance(term, Unary) and term.op is ops.neg:
            sign = -sign
            term = term.arg
        counts[term] += sign

    result = sum(term * count for term, count in counts.items()
                 if isinstance(term, (Number, Tensor)))
    for term, count in counts.items():
        if isinstance(term, (Number, Tensor)):
            continue
        if count == 0:
            continue
        if count == 1:
            result += term
        elif count == -1:
            result -= term
        else:
            result += count * term
    return result


__all__ = [
    'match_commutative',
    'simplify_sum',
    'try_match_reduction',
    'try_match_tensors',
]
