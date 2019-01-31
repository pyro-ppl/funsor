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


def log(x):
    return x.log()


def exp(x):
    return x.exp()


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

    def materialize(self):
        index = itertools.product(*map(domain, self.shape))
        tensor = torch.tensor([self[i] for i in index])
        tensor = tensor.reshape(self.shape)
        return TorchFunsor(self.dims, tensor)

    def log(self):
        return LazyFunsor(self.dims, self.shape, lambda *x: self[x].log())

    def exp(self):
        return LazyFunsor(self.dims, self.shape, lambda *x: self[x].exp())

    def __mul__(self, other):
        """
        Broadcasted pointwise multiplication.
        """
        if self.dims == other.dims:
            dims = self.dims
            shape = self.shape
        else:
            dims = tuple(sorted(set(self.dims + other.dims)))
            sizes = dict(zip(self.dims + other.dims, self.shape + other.shape))
            shape = tuple(sizes[d] for d in dims)

        def fn(*key):
            key1 = tuple(i for d, i in zip(dims, key) if d in self.dims)
            key2 = tuple(i for d, i in zip(dims, key) if d in other.dims)
            return self[key1] * other[key2]

        return LazyFunsor(dims, shape, fn)

    def mm(self, other):
        assert len(self.shape) == 2
        assert len(other.shape) == 2
        assert self.shape[-1] == other.shape[0]
        dims = (self.dims[0], other.dims[-1])
        shape = (self.shape[0], other.shape[-1])

        def fn(i, k):
            return sum(self[i, j] * other[j, k]
                       for j in domain(self.shape[-1]))

        return LazyFunsor(dims, shape, fn)


class LazyFunsor(Funsor):
    def __init__(self, dims, shape, fn):
        super(LazyFunsor, self).__init__(dims, shape)
        self.fn = fn

    def __getitem__(self, key):
        return self.fn(*key)


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

    def materialize(self):
        return self

    def log(self):
        return TorchFunsor(self.dims, self.tensor.log())

    def exp(self):
        return TorchFunsor(self.dims, self.tensor.exp())

    def __mul__(self, other):
        if isinstance(other, TorchFunsor):
            if self.dims == other.dims:
                return TorchFunsor(self.dims, self.tensor + other.tensor)
            dims = tuple(sorted(set(self.dims + other.dims)))
            equation = "{},{}->{}".format(self.dims, other.dims, dims)
            tensor = torch.einsum(equation, self.tensor, other.tensor)
            return TorchFunsor(dims, tensor)
        return super(TorchFunsor, self).__mul__(other)

    def mm(self, other):
        if isinstance(other, TorchFunsor):
            dims = (self.dims[0], other.dims[-1])
            tensor = self.tensor.mm(other.tensor)
            return TorchFunsor(dims, tensor)
        return super(TorchFunsor, self).mm(other)


class PolynomialFunsor(Funsor):
    def __init__(self, dims, shape, coefs):
        assert isinstance(coefs, dict)
        super(PolynomialFunsor, self).__init__(dims, shape)
        self.coefs = coefs

    def __getitem__(self, xs):
        xs = dict(zip(self.dims, xs))
        result = 0
        for key, value in self.coefs.items():
            term = value
            for dim in key:
                term = term * xs[dim]
            result = result + term
        return result


class TransformedFunsor(Funsor):
    def __init__(self, dims, shape, base_funsor,
                 pre_transforms=None, post_transform=None):
        super(TransformedFunsor, self).__init__(dims, shape)
        self.base_funsor = base_funsor
        if pre_transforms is None:
            pre_transforms = tuple(() for d in dims)
        if post_transform is None:
            post_transform = ()
        self.pre_transforms = pre_transforms
        self.post_transform = post_transform

    def __getitem__(self, key):
        key = list(key)
        for i, transform in enumerate(self.pre_transforms):
            for t in transform:
                key[i] = t(key[i])
        key = tuple(key)
        value = self.base_funsor[key]
        for t in self.post_transform:
            value = t(value)
        return value

    def log(self):
        post_transform = self.post_transforms + (log,)
        return TransformedFunsor(self.dims, self.shape, self.base_funsor,
                                 self.pre_transforms, post_transform)

    def exp(self):
        post_transform = self.post_transforms + (exp,)
        return TransformedFunsor(self.dims, self.shape, self.base_funsor,
                                 self.pre_transforms, post_transform)


def _lazy(fn, shape):
    args, vargs, kwargs, defaults = inspect.getargspec(fn)
    assert not vargs
    assert not kwargs
    dims = tuple(args)
    return LazyFunsor(dims, shape, fn)


def lazy(*shape):
    return functools.partial(_lazy, shape=shape)
