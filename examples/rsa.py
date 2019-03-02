from __future__ import absolute_import, division, print_function

import funsor
import funsor.ops as ops
from funsor.distributions import Bernoulli, Delta, Uniform
from funsor.domains import reals
from funsor.terms import Stack, Variable


def guess(p, value):

    @funsor.fix(reals(), reals())
    def fn(guess_, value):
        case_shallow = Uniform(0, 1, value=value)

        x = Variable('x', reals())
        mean = guess_(x).reduce(ops.logaddexp, 'x')
        case_deep = Delta(mean, value=value)

        cases = Stack((case_shallow, case_deep), 'deep')
        return (Bernoulli(p, value='deep') * cases).reduce(ops.logaddexp, 'deep')

    return fn(value)
