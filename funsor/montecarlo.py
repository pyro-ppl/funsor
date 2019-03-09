from __future__ import absolute_import, division, print_function

from multipledispatch import dispatch

import funsor.distributions as dist
import funsor.ops as ops
from funsor.gaussian import Gaussian
from funsor.ops import AssociativeOp
from funsor.registry import KeyedRegistry
from funsor.terms import Binary, Funsor, Reduce, eager
from funsor.torch import Tensor


def monte_carlo(cls, *args):
    result = _monte_carlo(cls, *args)
    if result is None:
        result = eager(cls, *args)
    return result


_monte_carlo = KeyedRegistry(lambda *args: None)
monte_carlo.register = _monte_carlo.register


@monte_carlo.register(Reduce, AssociativeOp, Funsor, frozenset)
def monte_carlo_reduce(op, arg, reduced_vars):
    if op is ops.logaddexp:
        return approximate(arg.cls, reduced_vars, *arg._ast_values).reduce(reduced_vars)

    return None  # return default implementation


################################################################################
# Approximation
################################################################################

approximate = KeyedRegistry(lambda *args: None)


# Try to sample from lhs.
@approximate.register(Binary, AssociativeOp, (Tensor, Gaussian), Funsor)
def approximate_binary(reduced_vars, op, lhs, rhs):
    if op is ops.add:
        return sample(lhs, reduced_vars) + rhs

    if op is ops.sub:
        return sample(lhs, reduced_vars) - rhs

    return None  # return default implementation


# Try to sample from rhs.
@approximate.register(Binary, AssociativeOp, Funsor, (Tensor, Gaussian))
def approximate_binary(reduced_vars, op, lhs, rhs):
    if op is ops.add:
        return lhs + sample(rhs, reduced_vars)

    return None  # return default implementation


# Try to recurse into components.
@approximate.register(Binary, AssociativeOp, Funsor, Funsor)
def approximate_binary(reduced_vars, op, lhs, rhs):
    if op is ops.add:
        approx_lhs = approximate(lhs)
        if approx_lhs is not None:
            return approx_lhs + rhs
        approx_rhs = approximate(rhs)
        if approx_rhs is not None:
            return lhs + approx_rhs

    if op is ops.sub:
        approx_lhs = approximate(lhs)
        if approx_lhs is not None:
            return approx_lhs - rhs

    return None  # return default implementation


################################################################################
# Samplers
################################################################################

@dispatch(dist.Distribution, frozenset)
def sample(x, reduced_vars):
    return x.sample(reduced_vars)


@dispatch(Tensor, frozenset)
def sample(x, reduced_vars):
    raise NotImplementedError('TODO return a product of deltas?')


@dispatch(Gaussian, frozenset)
def sample(x, reduced_vars):
    raise NotImplementedError('TODO return a product of deltas?')


__all__ = [
    'monte_carlo',
]
