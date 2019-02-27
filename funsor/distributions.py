from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import pyro.distributions as dist
from six import add_metaclass

import funsor.ops as ops
from funsor.domains import bint, reals
from funsor.terms import Funsor, FunsorMeta, Number, Variable, eager, to_funsor
from funsor.torch import Tensor, align_tensors, materialize


class Distribution(Funsor):
    """
    Funsor backed by a PyTorch distribution object.
    """
    dist_class = "defined by derived classes"

    def __init__(self, *args):
        params = tuple(zip(self._ast_fields, args))
        assert any(k == 'value' for k, v in params)
        inputs = OrderedDict()
        for name, value in params:
            assert isinstance(name, str)
            assert isinstance(value, Funsor)
            inputs.update(value.inputs)
        inputs = OrderedDict(inputs)
        output = reals()
        super(Distribution, self).__init__(inputs, output)
        self.params = params

    def __repr__(self):
        return '{}({})'.format(type(self).__name__,
                               ', '.join('{}={}'.format(*kv) for kv in self.params))

    def eager_subs(self, subs):
        assert isinstance(subs, tuple)
        if not any(k in self.inputs for k, v in subs):
            return self
        params = OrderedDict((k, v.eager_subs(subs)) for k, v in self.params)
        return type(self)(**params)

    def eager_reduce(self, op, reduced_vars):
        if op is ops.logaddexp and isinstance(self.value, Variable) and self.value.name in reduced_vars:
            return Number(0.)  # distributions are normalized
        return super(Distribution, self).reduce(op, reduced_vars)

    @classmethod
    def eager_log_prob(cls, **params):
        inputs, tensors = align_tensors(*params.values())
        params = dict(zip(params, tensors))
        value = params.pop('value')
        data = cls.dist_class(**params).log_prob(value)
        return Tensor(data, inputs)


################################################################################
# Distribution Wrappers
################################################################################

class CategoricalMeta(FunsorMeta):
    """
    Wrapper to fill in default params.
    """
    def __call__(cls, probs, value=None):
        return super(CategoricalMeta, cls).__call__(probs, value)


@add_metaclass(CategoricalMeta)
class Categorical(Distribution):
    dist_class = dist.Categorical

    def __init__(self, probs, value=None):
        probs = to_funsor(probs)
        if value is None:
            size = probs.output.shape[0]
            value = Variable('value', bint(size))
        else:
            value = to_funsor(value)
        super(Categorical, self).__init__(probs, value)


@eager.register(Categorical, Funsor, Number)
def eager_categorical(probs, value):
    return probs[value].log()


@eager.register(Categorical, (Number, Tensor), (Number, Tensor))
def eager_categorical(probs, value):
    return Categorical.eager_log_prob(probs=probs, value=value)


@eager.register(Categorical, (Number, Tensor), Variable)
def eager_categorical(probs, value):
    value = materialize(value)
    return Categorical.eager_log_prob(probs=probs, value=value)


class Normal(Distribution):
    dist_class = dist.Normal

    def __init__(self, loc, scale, value=None):
        loc = to_funsor(loc)
        scale = to_funsor(scale)
        if value is None:
            value = Variable('value', reals())
        else:
            value = to_funsor(value)
        super(Normal, self).__init__(loc, scale, value)


@eager.register(Normal, (Number, Tensor), (Number, Tensor), (Number, Tensor))
def eager_normal(loc, scale, value):
    return Normal.eager_log_prob(loc=loc, scale=scale, value=value)


__all__ = [
    'Categorical',
    'Distribution',
    'Normal',
]
