from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import torch

from funsor.core import Funsor, Tensor, to_funsor
import funsor.ops as ops


class Delta(Funsor):
    pass  # TODO


class Normal(Funsor):
    def __init__(self, loc, scale, log_normalizer=0.):
        self.loc = to_funsor(loc)
        self.scale = to_funsor(scale)
        self.log_normalizer = to_funsor(log_normalizer)
        schema = OrderedDict(value='real')
        for f in [self.loc, self.scale, self.log_normalizer]:
            assert 'value' not in f.dims
            schema.update(f.schema)
        dims = tuple(schema)
        shape = tuple(schema.values())
        super(Normal, self).__init__(dims, shape)

    def __call__(self, *args, **kwargs):
        kwargs.update(zip(self.dims, args))
        loc = self.loc(**kwargs)
        scale = self.scale(**kwargs)
        log_normalizer = self.log_normalizer(**kwargs)
        if (loc is self.loc and scale is self.scale and
                log_normalizer is self.log_normalizer):
            result = self
        else:
            result = Normal(loc, scale, log_normalizer)
        if 'value' not in kwargs:
            return result
        raise NotImplementedError('TODO')

    def reduce(self, op, dims):
        if op is ops.logaddexp:
            raise NotImplementedError('TODO')
        return super(Normal, self).reduce(op, dims)

    def argreduce(self, op, dims):
        if op is ops.sample:
            raise NotImplementedError('TODO')
        return super(Normal, self).argreduce(op, dims)

    def pointwise_binary(self, op, other):
        if op is ops.add:
            if isinstance(other, Tensor):
                raise NotImplementedError('TODO')
            if isinstance(other, Normal):
                raise NotImplementedError('TODO')
            if isinstance(other, Delta):
                raise NotImplementedError('TODO')
        return super(Normal, self).pointwise_binary(op, other)

    # TODO implement optimized .contract()
    # TODO implement optimized .argcontract()


class MultivariateNormal(Funsor):
    pass  # TODO


class ApproxNormal(Normal):
    r"""
    This implements EP of Gaussian mixtures,
    e.g. for switching linear dynamical systems.
    """
    def reduce(self, op, dims):
        if op is ops.logaddexp:
            if any(isinstance(self.schema[d], int) for d in dims):
                raise NotImplementedError('TODO match moments')
        return super(ApproxNormal, self).reduce(op, dims)


class TransformedDistribution(Funsor):
    def __init__(self, base, dim, transform):
        assert isinstance(base, Funsor)
        assert dim in base.dims
        assert isinstance(transform, torch.distributions.transforms.Transform)
        super(TransformedDistribution, self).__init__(base.dims, base.shape)
        self.base = base
        self.dim = dim
        self.transform = transform

    def __call__(self, *args, **kwargs):
        kwargs.update(zip(*args, **kwargs))
        if self.dim in kwargs:
            x = kwargs[self.dim]
            y = self.transform(x)
            kwargs[self.dim] = y
            jac = Tensor(x.dims, self.transform.log_abs_det_jacobian(x, y))
            return self.base(**kwargs) + jac
        base = self.base(**kwargs)
        if base is self.base:
            return self
        if self.dim in base:
            raise NotImplementedError('TODO handle collision')
        return TransformedDistribution(base, self.dim, self.transform)


__all__ = [
    'ApproxNormal',
    'Delta',
    'MultivariateNormal',
    'Normal',
    'TransformedDistribution',
]
