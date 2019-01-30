from __future__ import absolute_import, division, print_function

import functools
import inspect
import itertools

import torch

DOMAINS = ("real", "positive", "unit_interval")


def is_size(x):
    return isinstance(x, int) or x in DOMAINS


def domain(size):
    if isinstance(size, int):
        return range(size)
    raise NotImplementedError("cannot iterate over {}".format(domain))


class Funsor(object):
    def __init__(self, dims, shape):
        assert len(dims) == len(shape)
        assert all(isinstance(d, str) for d in dims)
        assert isinstance(shape, tuple)
        assert all(is_size(s) for s in shape)
        self.dims = dims
        self.shape = shape

    def size(self):
        return self.shape

    def mm(self, other):
        assert len(self.shape) == 2
        assert len(other.shape) == 2
        assert self.shape[-1] == other.shape[0]
        dims = (self.dims[0], other.dims[-1])
        shape = (self.shape[0], other.shape[-1])

        def fn(i, k):
            J = domain(self.shape[-1])
            return sum(self[i, j] * other[j, k] for j in J)

        return LazyFunsor(dims, shape, fn)


class TorchFunsor(Funsor):
    def __init__(self, dims, tensor):
        assert isinstance(tensor, torch.Tensor)
        assert len(dims) == tensor.dim()
        self.tensor = tensor
        super(TorchFunsor, self).__init__(dims, tensor.shape)

    def __getitem__(self, key):
        return self.tensor[key]

    def __setitem__(self, key, value):
        self.tensor[key] = value

    def mm(self, other):
        if isinstance(other, TorchFunsor):
            dims = (self.dims[0], other.dims[-1])
            tensor = self.tensor.mm(other.tensor)
            return TorchFunsor(dims, tensor)
        else:
            return super(TorchFunsor, self).mm(other)


class LazyFunsor(Funsor):
    def __init__(self, dims, shape, fn):
        super(LazyFunsor, self).__init__(dims, shape)
        self.fn = fn

    def __getitem__(self, key):
        return self.fn(*key)

    def materialize(self):
        index = itertools.product(*map(domain, self.shape))
        tensor = torch.tensor([self.fn(*i) for i in index])
        tensor = tensor.reshape(self.shape)
        return TorchFunsor(self.dims, tensor)


def _lazy(fn, shape):
    args, vargs, kwargs, defaults = inspect.getargspec(fn)
    assert not vargs
    assert not kwargs
    dims = tuple(args)
    return LazyFunsor(dims, shape, fn)


def lazy(*shape):
    return functools.partial(_lazy, shape=shape)
