from __future__ import absolute_import, division, print_function

import types

import torch
from multipledispatch import dispatch, Dispatcher
from six.moves import reduce

import funsor.distributions as dist
from funsor.handlers import Handler, Message, OpRegistry, effectful
from funsor.six import singledispatch
from funsor.terms import Binary, Finitary, Funsor, Number, Reduction, Substitution, Unary, Variable
from funsor.torch import Arange, Tensor


class EagerEval(OpRegistry):
    dispatcher = Dispatcher('EagerEval')


@EagerEval.register(Tensor)
def eager_tensor(dims, data):
    return Tensor(dims, data).materialize()  # .data


@EagerEval.register(Number)
def eager_number(data, dtype):
    return Number(data, dtype)


# TODO add general Normal
@EagerEval.register(dist.Normal)
def eager_distribution(loc, scale, value):
    return dist.Normal(loc, scale, value=value).materialize()


@EagerEval.register(Variable)
def eager_variable(name, size):
    if isinstance(size, int):
        return Arange(name, size)
    else:
        return Variable(name, size)


@EagerEval.register(Unary)
def eager_unary(op, v):
    return op(v)


@EagerEval.register(Substitution)
def eager_substitution(arg, subs):  # this is the key...
    return Substitution(arg, subs).materialize()


@EagerEval.register(Binary)
def eager_binary(op, lhs, rhs):
    return op(lhs, rhs)


@EagerEval.register(Finitary)
def eager_finitary(op, operands):
    if len(operands) == 1:
        return eager_unary(op, operands[0])  # XXX is this necessary?
    return reduce(op, operands[1:], operands[0])


@EagerEval.register(Reduction)
def eager_reduce(op, arg, reduce_dims):
    return arg.reduce(op, reduce_dims)


class TailCall(Message):
    pass


class trampoline(Handler):
    """Trampoline to handle tail recursion automatically"""
    def __enter__(self):
        self._schedule = []
        self._args_queue = []
        self._kwargs_queue = []
        self._returnvalue = None
        return super(trampoline, self).__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            while self._schedule:
                fn, nargs, nkwargs = self._schedule.pop(0)
                args = tuple(self._args_queue.pop(0) for i in range(nargs))
                kwargs = dict(self._kwargs_queue.pop(0) for i in range(nkwargs))
                self._args_queue.append(fn(*args, **kwargs))
            self._returnvalue = self._args_queue.pop(0)
            assert not self._args_queue and not self._kwargs_queue
        else:
            self._schedule, self._args_queue, self._kwargs_queue = [], [], []
            self._returnvalue = None
        return super(trampoline, self).__exit__(exc_type, exc_value, traceback)

    @dispatch(object)
    def process(self, msg):
        return super(trampoline, self).process(msg)

    @dispatch(TailCall)
    def process(self, msg):
        msg["stop"] = True  # defer until exit
        msg["value"] = True
        self._schedule.append((msg["fn"], len(msg["args"]), len(msg["kwargs"])))
        self._args_queue.extend(msg["args"])
        self._kwargs_queue.extend(list(msg["kwargs"].items()))
        return msg

    def __call__(self, *args, **kwargs):
        with self:
            self.fn(*args, **kwargs)
        return self._returnvalue


def _tail_call(fn, *args, **kwargs):
    """tail call annotation for trampoline interception"""
    return effectful(TailCall, fn)(*args, **kwargs)


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
def _eval_ground(x):
    return x


@_eval.register(tuple)
def _eval_tuple(x):
    return tuple(map(_eval, x))


@_eval.register(frozenset)
def _eval_frozenset(x):
    return frozenset(map(_eval, x))


__all__ = [
    'eval',
    'EagerEval',
]
