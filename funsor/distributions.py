from __future__ import absolute_import, division, print_function

from collections import OrderedDict, defaultdict

import torch.distributions as dist
from six import add_metaclass

import funsor.ops as ops
from funsor.adjoint import backward
from funsor.contract import contract
from funsor.terms import Funsor, Number, Tensor, Variable, align_tensors, to_funsor, ConsHashedMeta


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


DEFAULT_VALUE = Variable('value', 'real')


class DefaultValueMeta(ConsHashedMeta):
    def __call__(cls, *args, **kwargs):
        kwargs.setdefault('value', DEFAULT_VALUE)
        return super(DefaultValueMeta, cls).__call__(*args, **kwargs)


@add_metaclass(DefaultValueMeta)
class Distribution(Funsor):
    """
    Abstract base class for funsors representing univariate probability
    distributions over the leading dim.
    """
    def __init__(self, cls, value, **params):
        assert issubclass(cls, dist.Distribution)
        schema = OrderedDict([(value.name, value.shape[0])])
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
        self.value = value

    def __repr__(self):
        return '{}({}, value={})'.format(
            type(self).__name__,
            ', '.join('{}={}'.format(*kv) for kv in self.params.items()),
            self.value)

    def __call__(self, *args, **kwargs):
        kwargs = {d: to_funsor(v) for d, v in kwargs.items() if d in self.dims}
        kwargs.update(zip(self.dims, map(to_funsor, args)))
        value = kwargs.pop(self.value.name, None)
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
        if isinstance(value, Variable):
            return type(self)(value=value, **self.params)
        if isinstance(value, Tensor):
            if all(isinstance(v, (Number, Tensor)) for v in self.params.values()):
                dims, tensors = align_tensors(value, *self.params.values())
                value = tensors[0]
                params = dict(zip(self.params, tensors[1:]))
                data = self.cls(**params).log_prob(value)
                return Tensor(dims, data)
        return super(Distribution, self).__call__(value)

    def reduce(self, op, dims):
        if op is ops.logaddexp and self.value.name in dims:
            return Number(0.)  # distributions are normalized
        return super(Distribution, self).reduce(op, dims)

    # Legacy distributions interface:

    def log_prob(self, value, event_dims=()):
        # TODO handle event dims
        return self(value=value)

    def sample(self):
        return backward(ops.sample, self, frozenset(self.value.dims))

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
        return {term.value.name: Tensor(dims, data)}

    raise NotImplementedError


################################################################################
# Distribution Wrappers
################################################################################


class Beta(Distribution):
    def __init__(self, concentration1, concentration0, value=DEFAULT_VALUE):
        super(Beta, self).__init__(dist.Beta, value,
                                   concentration1=concentration1, concentration0=concentration0)


class Gamma(Distribution):
    def __init__(self, concentration, rate, value=DEFAULT_VALUE):
        super(Gamma, self).__init__(dist.Gamma, value,
                                    concentration=concentration, rate=rate)


class Binomial(Distribution):
    def __init__(self, total_count, probs, value=DEFAULT_VALUE):
        super(Binomial, self).__init__(dist.Binomial, value,
                                       total_count=total_count, probs=probs)


class Delta(Distribution):
    def __init__(self, v, log_density, value=DEFAULT_VALUE):
        super(Delta, self).__init__(dist.Delta, value, v=v, log_density=log_density)


class Normal(Distribution):
    def __init__(self, loc, scale, value=DEFAULT_VALUE):
        super(Normal, self).__init__(dist.Normal, value, loc=loc, scale=scale)

    def contract(self, sum_op, prod_op, other, dims):
        if sum_op is ops.logaddexp and prod_op is ops.add:
            d = self.value.name
            if (isinstance(other, Normal) and other.value.name not in self.dims and
                    d != other.value.name and d not in other.params['scale'].dims):
                for a0, a1 in match_affine(other.params['loc'], d):
                    loc1, scale1 = self.params['loc'], self.params['scale']
                    loc2, scale2 = other.params['loc'], other.params['scale']
                    loc = a0 + a1 * loc1 + loc2
                    scale = ((scale1 * a1) ** 2 + scale2 ** 2).sqrt()
                    return Normal(loc, scale)(other.value)
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
    'Distribution',
    'Gamma',
    'Normal',
]
