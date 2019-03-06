from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import torch
from six import add_metaclass

import funsor.ops as ops
from funsor.domains import reals
from funsor.terms import Binary, Funsor, FunsorMeta, Number, eager, to_funsor
from funsor.torch import Tensor, align_tensors


class DeltaMeta(FunsorMeta):
    """
    Wrapper to convert between OrderedDict and tuple and fill in defaults.
    """
    def __call__(cls, vs, log_density=0):
        if isinstance(vs, OrderedDict):
            vs = tuple(vs.items())
        assert isinstance(vs, tuple)

        # Convert log_density to a Tensor.
        log_density = to_funsor(log_density)
        if isinstance(log_density, Number):
            new_tensor = torch.tensor
            for name, x in vs:
                if isinstance(x, Tensor):
                    new_tensor = x.new_tensor
                    break
            log_density = Tensor(new_tensor(log_density.data))

        return super(DeltaMeta, cls).__call__(vs, log_density)


@add_metaclass(DeltaMeta)
class Delta(Funsor):
    def __init__(self, vs, log_density=0):
        assert isinstance(vs, tuple)
        assert isinstance(log_density, Funsor)
        assert log_density.output == reals()
        inputs = OrderedDict()
        for name, v in vs:
            assert isinstance(name, str)
            assert isinstance(v, Tensor)
            inputs[name] = v.output
            inputs.update(v.inputs)
        inputs.update(log_density.inputs)
        output = reals()
        super(Delta, self).__init__(inputs, output)
        self.vs = OrderedDict(vs)
        self.log_density = log_density

    def eager_subs(self, subs):
        delta_part = tuple((k, v) for k, v in subs if k in self.vs)
        index_part = tuple((k, v) for k, v in subs if k in self.inputs and k not in self.vs)
        if not (delta_part or index_part):
            return self

        vs = self.vs
        log_density = self.log_density
        if index_part:
            vs = OrderedDict((name, v.eager_subs(index_part)) for name, v in vs.items())
            log_density = log_density.eager_subs(index_part)

        if delta_part:
            vs = vs.copy()
            for name, value in delta_part:
                if not isinstance(value, Tensor):
                    raise ValueError
                v = vs.pop(name)
                assert v.output == value.output
                event_dim = len(v.output.shape)
                inputs, (v, log_density, value) = align_tensors(v, log_density, value)
                eq = (v == value)
                if event_dim:
                    eq = eq.reshape(eq.shape[:-event_dim] + (-1)).all(-1)
                data = eq.type(log_density.dtype).log() + log_density
                log_density = Tensor(data, inputs)

        return Delta(vs, log_density)

    def eager_reduce(self, op, reduced_vars):
        raise NotImplementedError('TODO')


@eager.register(Delta, tuple, Funsor)
def eager_delta(vs, log_density):
    if not vs:
        return log_density

    return None  # defer to default implementation


def _add_delta_funsor(delta, other):
    assert isinstance(delta, Delta)
    assert isinstance(other, Funsor)
    raise NotImplementedError('TODO')


@eager.register(Binary, object, Delta, Funsor)
def eager_binary(op, lhs, rhs):
    if op is ops.add:
        return _add_delta_funsor(lhs, rhs)

    return None  # defer to default implementation


@eager.register(Binary, object, Funsor, Delta)
def eager_binary(op, lhs, rhs):
    if op is ops.add:
        return _add_delta_funsor(rhs, lhs)

    return None  # defer to default implementation


@eager.register(Binary, object, Delta, Delta)
def eager_binary(op, lhs, rhs):
    log_density = lhs.log_density + rhs.log_density
    both = frozenset(lhs.vs).intersection(rhs.vs)
    if both:
        vs = OrderedDict((k, lhs.vs[k]) for k in both)
        subs = tuple((k, rhs.vs[k]) for k in both)
        log_density = Delta(vs, log_density).eager_subs(subs)
    vs = OrderedDict((k, v) for vs in (lhs.vs, rhs.vs) for k, v in vs.items() if k not in both)
    return Delta(vs, log_density)


__all__ = [
    'Delta',
]
