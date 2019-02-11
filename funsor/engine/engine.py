from __future__ import absolute_import, division, print_function

from six.moves import reduce

from funsor.handlers import effectful, Handler, Label, OpRegistry
from funsor.terms import Arange, Binary, Finitary, Funsor, Reduction, Substitution, Tensor, Unary, Variable


class EagerEval(OpRegistry):
    _terms_processed = {}
    _terms_postprocessed = {}


@EagerEval.register(Tensor)
def eager_tensor(dims, data):
    return Tensor(dims, data).materialize()  # .data


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
def eager_substitute(arg, subs):  # this is the key...
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
    assert isinstance(arg, Tensor)  # XXX is this actually true?
    return arg.reduce(op, reduce_dims)


class TailCall(Label):
    pass


class trampoline(Handler):
    """Trampoline to handle tail recursion automatically"""
    def __enter__(self):
        self._schedule = []
        self._returnvalue = None
        return super(trampoline, self).__enter__()

    def __exit__(self, *eargs):
        print(self._schedule)
        while self._schedule:
            fn, args, kwargs = self._schedule.pop(0)
            self._returnvalue = fn(*args, **kwargs)
        super(trampoline, self).__exit__(*eargs)

    def process(self, msg):
        if isinstance(msg["label"], TailCall):
            msg["stop"] = True  # defer until exit
            msg["value"] = True
            self._schedule.append((msg["fn"], msg["args"], msg["kwargs"]))
        return msg

    def __call__(self, *args, **kwargs):
        with self:
            self.fn(*args, **kwargs)
        return self._returnvalue


def _tail_call(fn, *args, **kwargs):
    """tail call annotation for trampoline interception"""
    return effectful(TailCall(), fn)(*args, **kwargs)


@trampoline
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

    if isinstance(x, Tensor):
        return _tail_call(effectful(Tensor, Tensor), x.dims, x.data)

    if isinstance(x, Variable):
        return _tail_call(effectful(Variable, Variable), x.name, x.shape[0])

    if isinstance(x, Substitution):
        return _tail_call(
            effectful(Substitution, Substitution),
            eval(x.arg),
            tuple((dim, eval(value)) for (dim, value) in x.subs)
        )

    # Arithmetic operations
    if isinstance(x, Unary):
        return _tail_call(effectful(Unary, Unary), x.op, eval(x.v))

    if isinstance(x, Binary):
        return _tail_call(effectful(Binary, Binary), x.op, eval(x.lhs), eval(x.rhs))

    if isinstance(x, Finitary):
        return _tail_call(effectful(Finitary, Finitary), x.op, tuple(eval(tx) for tx in x.operands))

    # Reductions
    if isinstance(x, Reduction):
        return _tail_call(effectful(Reduction, Reduction), x.op, eval(x.arg), x.reduce_dims)

    raise NotImplementedError


__all__ = [
    'eval',
    'EagerEval',
]
