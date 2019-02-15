from __future__ import absolute_import, division, print_function

from six.moves import reduce
from funsor.six import singledispatch
from multipledispatch import dispatch

import funsor.distributions as dist
from funsor.handlers import Handler, Message, OpRegistry, effectful
from funsor.terms import Binary, Finitary, Funsor, Number, Reduction, Substitution, Unary, Variable
from funsor.torch import Arange, Tensor


class EagerEval(OpRegistry):
    _terms_processed = {}
    _terms_postprocessed = {}


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


@singledispatch
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
    raise NotImplementedError


@eval.register(Tensor)
def _eval_tensor(x):
    return Tensor(x.dims, x.data)


@eval.register(dist.Distribution)
def _eval_distribution(x):
    return type(x)(**{k: eval(v) for k, v in x.params.items()})


@eval.register(Number)
def _eval_number(x):
    return Number(x.data, type(x.data))


@eval.register(Variable)
def _eval_variable(x):
    return Variable(x.name, x.shape[0])


@eval.register(Substitution)
def _eval_substitution(x):
    return Substitution(
        eval(x.arg),
        tuple((dim, eval(value)) for (dim, value) in x.subs)
    )


@eval.register(Unary)
def _eval_unary(x):
    return Unary(x.op, eval(x.v))


@eval.register(Binary)
def _eval_binary(x):
    return Binary(x.op, eval(x.lhs), eval(x.rhs))


@eval.register(Finitary)
def _eval_finitary(x):
    return Finitary(x.op, tuple(eval(tx) for tx in x.operands))


@eval.register(Reduction)
def _eval_reduction(x):
    return Reduction(x.op, eval(x.arg), x.reduce_dims)


__all__ = [
    'eval',
    'EagerEval',
]
