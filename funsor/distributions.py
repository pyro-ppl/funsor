from __future__ import absolute_import, division, print_function

from collections import OrderedDict, defaultdict

import pyro.distributions as dist

import funsor.ops as ops
from funsor.domains import ints, reals
from funsor.pattern import simplify_sum
from funsor.terms import Binary, Funsor, Number, Variable, eager, to_funsor
from funsor.torch import Tensor, align_tensors


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
    if dim not in a1.dims:
        a0 = expr(**{dim: 0.})
        yield a0, a1


class Distribution(Funsor):
    """
    Funsor backed by a PyTorch distribution object.
    """
    dist_class = "defined by derived classes"

    def __init__(self, *params):
        params = tuple(zip(self._ast_fields, params))
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

    def eager_reduce(self, op, dims):
        if op is ops.logaddexp and isinstance(self.value, Variable) and self.value.name in dims:
            return Number(0.)  # distributions are normalized
        return super(Distribution, self).reduce(op, dims)

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

class Categorical(Distribution):
    dist_class = dist.Categorical

    def __init__(self, probs=None, value=None):
        assert value is not None or probs is not None
        probs = to_funsor(probs) if probs is None else probs
        value = to_funsor(value) if value is None else value
        if probs is None:
            size = value.output.dtype
            probs = Variable('probs', reals(size))
        if value is None:
            size = probs.output.shape[0]
            value = Variable('value', ints(size))
        super(Categorical, self).__init__(probs, value)


class Normal(Distribution):
    dist_class = dist.Normal

    def __init__(self, loc=None, scale=None, value=None):
        loc = Variable('loc', reals()) if loc is None else to_funsor(loc)
        scale = Variable('scale', reals()) if scale is None else to_funsor(scale)
        value = Variable('value', reals()) if value is None else to_funsor(value)
        super(Normal, self).__init__(loc, scale, value)


@eager.register(Normal, (Number, Tensor), (Number, Tensor), (Number, Tensor))
def eager_normal(loc, scale, value):
    return Normal.eager_log_prob(loc=loc, scale=scale, value=value)


################################################################################
# Conjugacy Relationships
################################################################################

@eager.register(Binary, object, Normal, Normal)
def eager_binary_normal_normal(op, lhs, rhs):
    if op is not ops.add:
        return None  # defer to default implementation

    # Try updating prior from a ground observation.
    if op is ops.add and isinstance(lhs.value, Variable):
        d = lhs.value.name
        if (isinstance(rhs, Normal) and rhs.is_observed and
                d not in rhs.params['scale'].dims and d in rhs.params['loc'].dims):
            for a0, a1 in match_affine(rhs.params['loc'], d):
                print('UPDATE\n lhs = {}\n rhs = {}'.format(lhs, rhs))
                loc1, scale1 = lhs.params['loc'], lhs.params['scale']
                scale2 = rhs.params['scale']
                loc2 = (rhs.value - a0) / a1
                scale2 = scale2 / a1.abs()
                prec1 = scale1 ** -2
                prec2 = scale2 ** -2

                # Perform a filter update.
                prec3 = prec1 + prec2
                loc3 = loc1 + simplify_sum(loc2 - loc1) * (prec2 / prec3)
                scale3 = prec3 ** -0.5
                updated = Normal(loc3, scale3, value=lhs.value)
                # FIXME add log(a1) term?

                # Encapsulate the log_likelihood in a Normal for later pattern matching.
                prec4 = prec1 * prec2 / prec3
                scale4 = prec4 ** -0.5
                loc4 = simplify_sum(loc2 - loc1)
                log_likelihood = Normal(loc4, scale4)(value=0.) + a1.abs().log()  # FIXME
                result = updated + log_likelihood
                print(' result = {}'.format(result))
                return result

    return None  # defer to default implementation


__all__ = [
    'Categorical',
    'Distribution',
    'Normal',
]
