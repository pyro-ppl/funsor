from __future__ import absolute_import, division, print_function

from collections import OrderedDict

from six import add_metaclass

import funsor.ops as ops
from funsor.domains import reals
from funsor.ops import Op
from funsor.terms import Align, Binary, Funsor, FunsorMeta, Number, Variable, eager, to_funsor


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
    :param Funsor point: Value of the bound variable.
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
        index_part = tuple(index_part)
        if value is None and not index_part:
            return self

        name = self.name
        point = self.point.eager_subs(index_part)
        log_density = self.log_density.eager_subs(index_part)
        if value is not None:
            if isinstance(value, Variable):
                name = value.name
            elif not any(d.dtype == 'real' for side in (value, point) for d in side.inputs.values()):
                return (value == point).all().log() + log_density
            else:
                # TODO Compute a jacobian, update log_prob, and emit another Delta.
                raise ValueError('Cannot substitute a {} into a Delta'
                                 .format(type(value).__name__))
        return Delta(name, point, log_density)

    def eager_reduce(self, op, reduced_vars):
        if op is ops.logaddexp:
            if self.name in reduced_vars:
                return Number(0)  # Deltas are normalized.

        # TODO Implement ops.add to simulate .to_event().

        return None  # defer to default implementation

    def sample(self, sampled_vars):
        assert all(k == self.name for k in sampled_vars if k in self.inputs)
        return self


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
