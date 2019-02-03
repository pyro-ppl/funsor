from __future__ import absolute_import, division, print_function

import math
from collections import OrderedDict

import torch

import funsor.ops as ops
from funsor.terms import Funsor, Tensor, Variable, align_tensors, to_funsor, var


def _drop_dims(dist, dims):
    """
    Drop dims from a distribution.

    This is typically needed when a distribution is morally constant wrt some
    dims, but tracks them to determine sample shape. In this case, slicing or
    reducing them simply removes names from tracking.
    """
    schema = OrderedDict([s for s in dist.schema.items() if s[0] not in dims])
    if schema == dist.schema:
        return dist
    dims = tuple(schema.keys())
    shape = tuple(schema.values())
    return type(dist)(dims[1:], shape[1:])


class Distribution(Funsor):
    """
    Base class for funsors representing non-normalized univariate probability
    distributions over the leading dim.

    Derived classes may implement pairs of methods, ``._xxx_param()`` to
    operate on parameters, and ``._xxx_value()`` to operate on the random
    variable.
    """
    def __init__(self, dims, shape):
        assert dims
        super(Distribution, self).__init__(dims, shape)

    @property
    def batch_dims(self):
        return self.dims[1:]

    @property
    def batch_shape(self):
        return self.shape[1:]

    def _split_kwargs(self, args, kwargs):
        kwargs.update(zip(self.dims, args))
        value = kwargs.pop(self.dims[0], None)
        return kwargs, value

    def _split_dims(self, dims):
        value = (self.dims[0] in dims)
        dims = frozenset(dims) - self.dims[0]
        return dims, value

    def __call__(self, *args, **kwargs):
        kwargs, value = self._split_kwargs(args, kwargs)
        result = self
        if kwargs:
            result = result._call_param(kwargs)
        if value is not None:
            result = result._call_value(value)
        return result

    def _call_param(self, kwargs):
        raise NotImplementedError

    def _call_value(self, value):
        if isinstance(value, Variable):
            raise NotImplementedError('TODO create transformed distribution')
        raise NotImplementedError

    def reduce(self, op, dims):
        dims, value = self._split_dims(dims)
        result = self
        if dims:
            result = result._reduce_param(op, dims)
        if value is not None:
            result = result._reduce_value(op)
        return result

    def _reduce_param(self, op, dims):
        raise NotImplementedError

    def _reduce_value(self, op):
        raise NotImplementedError


class Delta(Distribution):
    def __init__(self, value, name='value'):
        assert isinstance(value, Funsor)
        dims = (name,) + value.dims
        shape = ('real',) + value.shape
        super(Delta, self).__init__(dims, shape)
        self.value = value

    def _call_param(self, kwargs):
        value = self.value(**kwargs)
        if value is self.value:
            return self
        return type(self)(value)

    def _call_value(self, value):
        # Lazy evaluation.
        if isinstance(value, Variable):
            assert value.shape[0] == 'real'
            value = value.dims[0]
            # Fall through to str case.
        if isinstance(value, str):
            if value == self.dims[0]:
                return self
            return type(self).__init__(self.batch_dims, self.batch_shape, name=value)

        # Eager evaluation.
        if isinstance(value, Tensor) and isinstance(self.value, Tensor):
            dims, (lhs, rhs) = align_tensors(value, self.value)
            return Tensor(dims, (lhs == rhs).type_as(lhs))
        if isinstance(value, Variable) and value is self.value:
            return to_funsor(0.)

        raise NotImplementedError('TODO support laziness')


class StandardNormal(Distribution):
    def __init__(self, batch_dims=(), batch_shape=(), name='value'):
        assert name not in batch_dims
        dims = (name,) + batch_dims
        shape = ('real',) + batch_shape
        super(StandardNormal, self).__init__(dims, shape)

    def _call_param(self, kwargs):
        return _drop_dims(self, kwargs)

    def _call_value(self, value):
        # Lazy evaluation.
        if isinstance(value, Variable):
            assert value.shape[0] == 'real'
            value = value.dims[0]
            # Fall through to str case.
        if isinstance(value, str):
            if value == self.dims[0]:
                return self
            return type(self).__init__(self.batch_dims, self.batch_shape, name=value)

        # Eager evaluation.
        if isinstance(value, Tensor):
            log_prob = -0.5 * value.data ** 2 - math.log(math.sqrt(2 * math.pi))
            return Tensor(value.dims, log_prob)

        return super(StandardNormal, self)._call_value(value)

    def _reduce_param(self, op, dims):
        if op is ops.add:
            return _drop_dims(self, dims)
        raise NotImplementedError

    def _reduce_value(self, op):
        if op is ops.logaddexp:
            return to_funsor(0.)
        raise NotImplementedError

    def _argreduce_value(self, op):
        if op is ops.sample:
            value = Tensor(self.dims[1:], torch.randn(self.shape[1:]))
            return {self.dims[0]: value}, to_funsor(0.)
        raise NotImplementedError

    def binary(self, op, other):
        if op is ops.add:
            if isinstance(other, Normal):
                raise NotImplementedError('TODO')
            if isinstance(other, Tensor):
                raise NotImplementedError('TODO')
            if isinstance(other, Delta):
                raise NotImplementedError('TODO')
        return super(Normal, self).binary(op, other)


def Normal(loc, scale, name='value'):
    loc = to_funsor(loc)
    scale = to_funsor(scale)
    schema = OrderedDict([(name, 'real')])
    schema.update(loc.schema)
    schema.update(scale.schema)
    dims = tuple(schema)
    shape = tuple(schema.values())
    std = StandardNormal(dims[1:], shape[1:])
    return std((var(name, 'real') - loc) / scale)


__all__ = [
    'Delta',
    'Distribution',
    'Normal',
    'StandardNormal',
]
