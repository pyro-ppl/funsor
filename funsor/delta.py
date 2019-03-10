from __future__ import absolute_import, division, print_function

from collections import OrderedDict

from six import add_metaclass

import funsor.ops as ops
from funsor.domains import reals
from funsor.ops import Op
from funsor.terms import Align, Binary, Funsor, FunsorMeta, Number, Variable, eager, to_funsor
from funsor.torch import Tensor


class DeltaMeta(FunsorMeta):
    """
    Wrapper to convert point to a funsor.
    """
    def __call__(cls, name, point):
        point = to_funsor(point)
        return super(DeltaMeta, cls).__call__(name, point)


@add_metaclass(DeltaMeta)
class Delta(Funsor):
    """
    Normalized delta distribution binding a single variable.

    :param str name: Name of the bound variable.
    :param Funsor point: Value of the bound variable.
    """
    def __init__(self, name, point):
        assert isinstance(name, str)
        assert isinstance(point, Funsor)
        inputs = OrderedDict([(name, point.output)])
        inputs.update(point.inputs)
        output = reals()
        super(Delta, self).__init__(inputs, output)
        self.name = name
        self.point = point

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
        index_part = tuple(index_part)
        if value is None and not index_part:
            return self

        name = self.name
        point = self.point.eager_subs(index_part)
        if value is not None:
            if isinstance(value, Variable):
                name = value.name
            elif isinstance(value, (Number, Tensor)) and isinstance(point, (Number, Tensor)):
                return (value == point).all().log()
            else:
                # TODO Compute a jacobian, update log_prob, and emit another Delta.
                raise ValueError('Cannot substitute a {} into a Delta'
                                 .format(type(value).__name__))
        return Delta(name, point)

    def eager_reduce(self, op, reduced_vars):
        if op is ops.logaddexp:
            if self.name in reduced_vars:
                return Number(0)  # Deltas are normalized.

        # TODO Implement ops.add to simulate .to_event().

        return None  # defer to default implementation


@eager.register(Binary, Op, Delta, (Funsor, Delta, Align))
def eager_binary(op, lhs, rhs):
    if op is ops.add or op is ops.sub:
        if lhs.name in rhs.inputs:
            rhs = rhs(**{lhs.name: lhs.point})
            return op(lhs, rhs)

    return None  # defer to default implementation


@eager.register(Binary, Op, (Funsor, Align), Delta)
def eager_binary(op, lhs, rhs):
    if op is ops.add:
        if rhs.name in lhs.inputs:
            lhs = lhs(**{rhs.name: rhs.point})
            return op(lhs, rhs)

    return None  # defer to default implementation


__all__ = [
    'Delta',
]
