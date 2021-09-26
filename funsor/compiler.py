# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools

import funsor

from .cnf import Contraction
from .ops.program import OpProgram, make_tuple
from .tensor import Tensor
from .terms import Binary, Funsor, Number, Tuple, Unary, Variable


def compile_funsor(expr: Funsor) -> OpProgram:
    """
    Compiles a symbolic :class:`~funsor.terms.Funsor` to an
    :class:`~funsor.ops.program.OpProgram` that runs on backend values.

    Example::

        # Create a lazy expression.
        a = Variable("a", Reals[3, 3])
        b = Variable("b", Reals[3])
        x = Variable("x", Reals[3])
        expr = a @ x + b

        # Evaluate via Funsor substitution.
        data = dict(a=randn(3, 3), b=randn(3), x=randn(3))
        expected = expr(**data).data

        # Alternatively evaluate via a program.
        program = compile_funsor(expr)
        actual = program(**data)
        assert (acutal == expected).all()

    :param Funsor expr: A funsor expression to evaluate.
    :returns: An op program.
    :rtype: ~funsor.ops.program.OpProgram
    """
    assert isinstance(expr, Funsor)

    # Lower and convert to A-normal form.
    lowered_expr = lower(expr)
    anf = list(funsor.interpreter.anf(lowered_expr))
    ids = {}

    # Collect constants (leaves).
    constants = []
    for f in anf:
        if isinstance(f, (Number, Tensor)):
            ids[f] = len(ids)
            constants.append(f.data)

    # Collect input variables (leaves).
    inputs = []
    for k, d in expr.inputs.items():
        f = Variable(k, d)
        ids[f] = len(ids)
        inputs.append(k)

    # Collect operations to be computed (internal nodes).
    operations = []
    for f in anf:
        if f in ids:
            continue  # constant or free variable
        ids[f] = len(ids)
        if isinstance(f, Unary):
            arg_ids = (ids[f.arg],)
            operations.append((f.op, arg_ids))
        elif isinstance(f, Binary):
            arg_ids = (ids[f.lhs], ids[f.rhs])
            operations.append((f.op, arg_ids))
        elif isinstance(f, Tuple):
            arg_ids = tuple(ids[arg] for arg in f.args)
            operations.append((make_tuple, arg_ids))
        elif isinstance(f, tuple):
            continue  # Skip from Tuple directly to its elements.
        else:
            raise NotImplementedError(type(f).__name__)

    return OpProgram(constants, inputs, operations)


def lower(expr: Funsor) -> Funsor:
    """
    Lower a funsor expression:
    - eliminate bound variables
    - convert Contraction to Binary

    :param Funsor expr: An arbitrary funsor expression.
    :returns: A lowered funsor expression.
    :rtype: Funsor
    """
    # FIXME should this be lazy? What about Lambda?
    with funsor.interpretations.reflect:
        return _lower(expr)


@functools.singledispatch
def _lower(x):
    raise NotImplementedError(type(x).__name__)


@_lower.register(Number)
@_lower.register(Tensor)
@_lower.register(Variable)
def _lower_atom(x):
    return x


@_lower.register(Tuple)
def _lower_tuple(x):
    args = tuple(_lower(arg) for arg in x.args)
    return Tuple(args)


@_lower.register(Unary)
def _lower_unary(x):
    arg = _lower(x.arg)
    return Unary(x.op, arg)


@_lower.register(Binary)
def _lower_binary(x):
    lhs = _lower(x.lhs)
    rhs = _lower(x.rhs)
    return Binary(x.op, lhs, rhs)


@_lower.register(Contraction)
def _lower_contraction(x):
    if x.reduced_vars:
        raise NotImplementedError("TODO")

    terms = [_lower(term) for term in x.terms]
    bin_op = functools.partial(Binary, x.bin_op)
    return functools.reduce(bin_op, terms)


__all__ = [
    "lower",
]
