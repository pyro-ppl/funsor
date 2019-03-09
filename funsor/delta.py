from __future__ import absolute_import, division, print_function

from collections import OrderedDict

from six import add_metaclass

import funsor.ops as ops
from funsor.ops import Op
from funsor.domains import reals
from funsor.terms import Binary, Funsor, FunsorMeta, Number, eager, to_funsor


class DeltaMeta(FunsorMeta):
    """
    Wrapper to fill in defaults.
    """
    def __call__(cls, name, point, log_density=0):
        point = to_funsor(point)
        log_density = to_funsor(log_density)
        return super(DeltaMeta, cls).__call__(name, point, log_density)


@add_metaclass(DeltaMeta)
class Delta(Funsor):
    """
    Normalized delta distribution binding a single variable.

    :param str name: Name of the bound variable.
    :param Funsor value: Value of the bound variable.
    :param Funsor log_density: Optional log density to be added when evaluating
        at a point. This is needed to make :class:`Delta` closed under
        differentiable substitution.
    """
    def __init__(self, name, point, log_density=0):
        assert isinstance(name, str)
        assert isinstance(point, Funsor)
        assert isinstance(log_density, Funsor)
        assert log_density.output == reals()
        inputs = OrderedDict([(name, point.output)])
        inputs.update(point.inputs)
        inputs.update(log_density.inputs)
        output = reals()
        super(Delta, self).__init__(inputs, output)
        self.name = name
        self.point = point
        self.log_density = log_density

    def eager_subs(self, subs):
        value = None
        index_part = []
        for k, v in subs:
            if k in self.inputs:
                if k == self.name:
                    value = v
                else:
                    assert self.name not in v.inputs
                    index_part.append((k, v))
        if value is None and not index_part:
            return self

        point = self.point.eager_subs(index_part)
        log_density = self.log_density.eager_subs(index_part)
        if value is not None:
            # TODO only use (==).all() on ground values.
            # On other values compute a jacobian and emit another Delta.
            # FIXME this requires change to .all() behavior.
            # FIXME this requires type promotion for boolean.log().
            return (value == point).all().log() + log_density
        return Delta(self.name, point, log_density)

    def eager_reduce(self, op, reduced_vars):
        if op is ops.logaddexp:
            if self.name in reduced_vars:
                # FIXME this requires .expand() operation.
                return Number(0).expand(frozenset(self.inputs) - reduced_vars)

        return None  # defer to default implementation


@eager.register(Binary, Op, Delta, Funsor)
def eager_binary(op, lhs, rhs):
    if op is ops.add or op is ops.sub:
        if lhs.name in rhs.inputs:
            rhs = rhs(**{lhs.name, lhs.value})
            return op(lhs, rhs)

    return None  # defer to default implementation


@eager.register(Binary, Op, Funsor, Delta)
def eager_binary(op, lhs, rhs):
    if op is ops.add:
        if rhs.name in lhs.inputs:
            lhs = lhs(**{rhs.name, rhs.value})
            return op(lhs, rhs)

    return None  # defer to default implementation


__all__ = [
    'Delta',
]
