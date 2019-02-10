from __future__ import absolute_import, division, print_function

from collections import OrderedDict, defaultdict

import torch.distributions as dist

import funsor.ops as ops
from funsor.adjoint import backward
from funsor.terms import Funsor, Number, Tensor, align_tensors, to_funsor


def log_abs_det(jacobian):
    result = 0.
    for i, row in jacobian.items():
        for j, entry in row.items():
            if i != j:
                raise NotImplementedError('TODO handle non-diagonal jacobians')
                result += ops.log(entry)
    return result


def log_abs_det_jacobian(schema, transform):
    jacobian = defaultdict(dict)
    for key, value in transform.items():
        for dim in value.dims:
            jacobian[key][dim] = value.grad(dim)
    return log_abs_det(jacobian)


class Distribution(Funsor):
    """
    Abstract base class for funsors representing univariate probability
    distributions over the leading dim, which is named 'value'.
    """
    def __init__(self, cls, **params):
        assert issubclass(cls, dist.Distribution)
        schema = OrderedDict([('value', 'real')])
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
            type(self).__name__,
            ', '.join('{}={}'.format(*kv) for kv in self.params.items()))

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
        params = {k: v(**kwargs) for k, v in self.params.items()}
        return type(self)(**params)

    def _call_value(self, value):
        if isinstance(value, Tensor):
            if all(isinstance(v, (Number, Tensor)) for v in self.params.values()):
                dims, tensors = align_tensors(value, *self.params.values())
                value = tensors[0]
                params = dict(zip(self.params, tensors[1:]))
                data = self.cls(**params).log_prob(value)
                return Tensor(dims, data)
        return super(Distribution, self).__call__(value)

    def reduce(self, op, dims):
        if op is ops.logaddexp and 'value' in dims:
            return Number(0.)  # distributions are normalized
        return super(Distribution, self).reduce(op, dims)

    # Legacy distributions interface:

    def log_prob(self, value):
        return self(value=value)

    def sample(self):
        return backward(ops.sample, self, frozenset('value'))


@backward.register(ops.sample, Distribution)
def _sample_torch_distribution(term, dims):
    if len(dims) != 1:
        raise NotImplementedError('TODO')
    if 'value' not in dims:
        raise NotImplementedError('TODO')

    if all(isinstance(v, (Number, Tensor)) for v in term.params.values()):
        dims, tensors = align_tensors(*term.params.values())
        params = dict(zip(term.params, tensors))
        data = term.cls(**params).rsample()
        return {'value': Tensor(dims, data)}

    raise NotImplementedError


################################################################################
# Distribution Wrappers
################################################################################

class Normal(Distribution):
    def __init__(self, loc, scale):
        super(Normal, self).__init__(dist.Normal, loc=loc, scale=scale)


class LogNormal(Distribution):
    def __init__(self, loc, scale):
        super(LogNormal, self).__init__(dist.LogNormal, loc=loc, scale=scale)


class Gamma(Distribution):
    def __init__(self, concentration, rate):
        super(Gamma, self).__init__(dist.Gamma, concentration=concentration, rate=rate)


################################################################################
# Conjugacy Relationships
################################################################################

# TODO

__all__ = [
    'Distribution',
    'Gamma',
    'LogNormal',
    'Normal',
]
