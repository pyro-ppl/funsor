from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import torch.distributions as dist

import funsor.ops as ops
from funsor.terms import Funsor, Number, Tensor, align_tensors, to_funsor


class Distribution(Funsor):
    """
    Abstract base class for funsors representing univariate probability
    distributions over the leading dim, which is named 'value'.
    """
    def __init__(self, cls, **params):
        assert issubclass(cls, dist.Distribution)
        schema = OrderedDict([('value', 'density')])
        self.params = OrderedDict()
        for k, v in sorted(params.items()):
            assert isinstance(k, str)
            v = to_funsor(v)
            schema.update(v.schema)
            self.params[k] = v
        dims = tuple(schema)
        shape = tuple(schema.values())
        super(Distribution, self).__init__(dims, shape)
        self.cls = cls

    def __repr__(self):
        return '{}({})'.format(
            self.cls.__name__,
            ', '.join(map('{}={}'.format, self.params.items())))

    def __call__(self, *args, **kwargs):
        kwargs = {d: to_funsor(v) for d, v in kwargs.items() if d in self.dims}
        kwargs.update(zip(self.dims, map(to_funsor, args)))
        value = kwargs.pop('value', None)
        result = self
        if kwargs:
            result = result._call_param(kwargs)
        if value is not None:
            result = result._call_value(value)
        return result

    def _call_param(self, kwargs):
        params = frozenset((k, v(**kwargs)) for k, v in self.params.items())
        return Distribution(self.cls, params)

    def _call_value(self, value):
        if isinstance(value, Tensor):
            if all(isinstance(v, (Number, Tensor)) for v in self.params.values()):
                dims, tensors = align_tensors(value, *self.params.values())
                value = tensors[0]
                params = dict(zip(self.params, tensors[1:]))
                data = self.cls(**params).log_prob(value)
                return Tensor(dims, data)
        return super(Distribution, self).__call__(value)

    def argreduce(self, op, dims):
        if op is ops.sample:
            if isinstance(dims, str):
                dims = (dims,)
            if set(dims).intersection(self.dims) == {'value'}:
                if all(isinstance(v, Tensor) for v in self.params.values()):
                    dims, tensors = align_tensors(*self.params.values())
                    params = dict(zip(self.params, tensors))
                    data = self.cls(**params).rsample()
                    return Tensor(dims, data)
        return super(Distribution, self).argreduce(op, dims)


class Normal(Distribution):
    def __init__(self, loc, scale):
        super(Normal, self).__init__(dist.Normal, loc=loc, scale=scale)


class LogNormal(Distribution):
    def __init__(self, loc, scale):
        super(LogNormal, self).__init__(dist.LogNormal, loc=loc, scale=scale)


class Gamma(Distribution):
    def __init__(self, concentration, rate):
        super(Gamma, self).__init__(dist.Gamma, concentration=concentration, rate=rate)


__all__ = [
    'Distribution',
    'Gamma',
    'LogNormal',
    'Normal',
]
