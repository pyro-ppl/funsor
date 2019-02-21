from __future__ import absolute_import, division, print_function

import types
from collections import OrderedDict

import torch

from funsor.six import singledispatch
from funsor.terms import Funsor
from funsor.domains import Domain


def eval(x):
    r"""
    Overloaded partial evaluation of deferred expression.
    Default semantics: do nothing (reflect)

    This handles a limited class of expressions, raising
    ``NotImplementedError`` in unhandled cases.

    :param Funsor x: An input funsor, typically deferred.
    :return: An evaluated funsor.
    :rtype: Funsor
    :raises: NotImplementedError
    """
    assert isinstance(x, Funsor)
    return _eval(x)


@singledispatch
def _eval(x):
    raise ValueError(type(x))


@_eval.register(Funsor)
def _eval_funsor(x):
    return type(x)(*map(_eval, x._ast_values))


@_eval.register(str)
@_eval.register(int)
@_eval.register(float)
@_eval.register(type)
@_eval.register(types.FunctionType)
@_eval.register(types.BuiltinFunctionType)
@_eval.register(torch.Tensor)
@_eval.register(Domain)
def _eval_ground(x):
    return x


@_eval.register(tuple)
def _eval_tuple(x):
    return tuple(map(_eval, x))


@_eval.register(frozenset)
def _eval_frozenset(x):
    return frozenset(map(_eval, x))


@_eval.register(dict)
def _eval_dict(x):
    return {key: _eval(value) for key, value in x.items()}


@_eval.register(OrderedDict)
def _eval_ordereddict(x):
    return OrderedDict((key, _eval(value)) for key, value in x.items())


__all__ = [
    'eval',
]
