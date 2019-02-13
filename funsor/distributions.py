from __future__ import absolute_import, division, print_function

from collections import OrderedDict, defaultdict

import torch.distributions as dist

import funsor.ops as ops
from funsor.adjoint import backward
from funsor.contract import contract
from funsor.terms import Funsor, Number, Tensor, align_tensors, to_funsor


def log_abs_det(jacobian):
    result = 0.
    for i, row in jacobian.items():
        for j, entry in row.items():
            if i != j:
                raise NotImplementedError('TODO handle non-diagonal jacobians')
                result += ops.log(entry)
    return result


def log_abs_det_jacobian(transform):
    jacobian = defaultdict(dict)
    for key, value in transform.items():
        for dim in value.dims:
            jacobian[key][dim] = value.jacobian(dim)
    return log_abs_det(jacobian)


def match_affine(expr, dim):
    assert isinstance(expr, Funsor)
    assert isinstance(dim, str)
    a1 = expr.jacobian(dim)
    if dim not in a1:
        a0 = expr(d=0.)
        yield a0, a1


# WIP candidate base distribution interface
class AbstractDistribution(object):
    def __init__(self, samples, log_prob):
        assert isinstance(samples, dict)
        assert isinstance(log_prob, Funsor)
        for k, v in samples.items():
            assert isinstance(k, str)
            assert isinstance(v, Funsor)
            assert set(v.dims) == set(log_prob.dims)
        super(AbstractDistribution, self).__init__()
        self.samples = samples
        self.log_prob = log_prob

    @property
    def dims(self):
        return self.log_prob.dims

    def sample(self, dims):
        if not dims:
            return self
        log_prob = self.log_prob.reduce(ops.sample, dims)
        samples = backward(ops.sample, log_prob)
        samples.update((k, v(**samples)) for k, v in self.samples)
        return AbstractDistribution(samples, log_prob)


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

    def log_prob(self, value, event_dims=()):
        # TODO handle event dims
        return self(value=value)

    def sample(self):
        return backward(ops.sample, self, frozenset('value'))

    def transform(self, **transform):
        return self(**transform) + log_abs_det_jacobian(transform)  # sign error?


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

class Beta(Distribution):
    def __init__(self, concentration1, concentration0):
        super(Beta, self).__init__(dist.Beta, concentration1=concentration1, concentration0=concentration0)


class Binomial(Distribution):
    def __init__(self, total_count, probs):
        super(Binomial, self).__init__(dist.Binomial, total_count=total_count, probs=probs)


class Delta(Distribution):
    def __init__(self, v, log_density=0.):
        super(Delta, self).__init__(v=v, log_density=log_density)


class Dirichlet(Distribution):
    def __init__(self, concentration):
        super(Distribution, self).__init__(dist.Dirichlet, concentration=concentration)


class Gamma(Distribution):
    def __init__(self, concentration, rate):
        super(Gamma, self).__init__(dist.Gamma, concentration=concentration, rate=rate)


class LogNormal(Distribution):
    def __init__(self, loc, scale):
        super(LogNormal, self).__init__(dist.LogNormal, loc=loc, scale=scale)


class Multinomial(Distribution):
    def __init__(self, total_count, probs):
        super(Multinomial, self).__init__(dist.Multinomial, total_count=total_count, probs=probs)


class Normal(Distribution):
    def __init__(self, loc, scale):
        super(Normal, self).__init__(dist.Normal, loc=loc, scale=scale)

    def contract(self, sum_op, prod_op, other, dims):
        if sum_op is ops.logaddexp and prod_op is ops.add:
            d = self.value_dim
            if (isinstance(other, Normal) and other.value_dim not in self.dims and
                    d != other.value_dim and d not in other.params['scale'].dims):
                for a0, a1 in match_affine(other.params['loc'], d):
                    loc1, scale1 = self.params['loc'], self.params['scale']
                    loc2, scale2 = other.params['loc'], other.params['scale']
                    loc = a0 + a1 * loc1 + loc2
                    scale = ((scale1 * a1) ** 2 + scale2 ** 2).sqrt()
                    return Normal(loc, scale)(other.value_dim)
        return super(Normal, self).contract(sum_op, prod_op, other, dims)


################################################################################
# Conjugacy Relationships
################################################################################

@contract.register((ops.logaddexp, ops.mul), Delta, Funsor)
@contract.register((ops.logaddexp, ops.mul), Delta, Delta)
def _contract_delta(lhs, rhs):
    return rhs(value=lhs.params['v'])


@contract.register((ops.logaddexp, ops.mul), Normal, Normal)
def _contract_normal_normal(lhs, rhs, reduce_dims):
    raise NotImplementedError('TODO')


__all__ = [
    'Beta',
    'Binomial',
    'Delta',
    'Dirichlet',
    'Distribution',
    'Gamma',
    'LogNormal',
    'Multinomial',
    'Normal',
]
