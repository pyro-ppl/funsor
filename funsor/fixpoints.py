from __future__ import absolute_import, division, print_function

import contextlib
import functools
from collections import OrderedDict, defaultdict

import torch

from funsor.domains import Domain
from funsor.registry import KeyedRegistry
from funsor.six import getargspec
from funsor.terms import Funsor, Variable, eager, reflect
from funsor.torch import Tensor

MAX_TRUNCATION_DEPTH = 10
TRUNCATION_DEPTHS = defaultdict(int)


@contextlib.contextmanager
def truncate(fn, output):
    TRUNCATION_DEPTHS[fn] += 1
    if TRUNCATION_DEPTHS[fn] <= MAX_TRUNCATION_DEPTH:
        result = fn
    else:
        def result(*args, **kwargs):
            return Tensor(torch.tensor(float('nan')).expand(output.shape))
    try:
        yield result
    finally:
        TRUNCATION_DEPTHS[fn] -= 1


_truncated = KeyedRegistry(default=lambda *args: None)


def truncated(cls, *args):
    result = _truncated(cls, *args)
    if result is None:
        result = eager(cls, *args)
    if result is None:
        result = reflect(cls, *args)
    return result


truncated.register = _truncated.register


class Fix(Funsor):
    def __init__(self, fn, output, args):
        assert callable(fn)
        assert isinstance(args, tuple)
        inputs = OrderedDict()
        for arg in args:
            assert isinstance(arg, Funsor)
            inputs.update(arg.inputs)
        super(Fix, self).__init__(inputs, output)
        self.fn = fn
        self.args = args

    def __repr__(self):
        return 'Fix({})'.format(', '.join(
            [type(self).__name__, repr(self.output)] + list(map(repr, self.args))))

    def __str__(self):
        return 'Fix({})'.format(', '.join(
            [type(self).__name__, str(self.output)] + list(map(str, self.args))))

    def eager_subs(self, subs):
        if not any(k in self.inputs for k, v in subs):
            return self
        args = tuple(arg.eager_subs(subs) for arg in self.args)
        return Fix(self.fn, self.output, args)


@truncated.register(Fix, object, Domain, tuple)
def eager_function(fn, output, args):
    with truncate(fn) as t:
        return fn(t, *args)


def _fix(inputs, output, fn):
    names = getargspec(fn)[0]
    names = names[1:]  # Assume initial name is fn itself.
    args = tuple(Variable(name, domain) for (name, domain) in zip(names, inputs))
    assert len(args) == len(inputs)
    return Fix(fn, output, args)


def fix(*signature):
    assert signature
    assert all(isinstance(d, Domain) for d in signature)
    inputs, output = signature[:-1], signature[-1]
    return functools.partial(_fix, inputs, output)


__all__ = [
    'Fix',
    'fix',
    'truncate',
    'truncated',
]
