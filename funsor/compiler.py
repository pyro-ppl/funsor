# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools

import funsor

from .cnf import Contraction
from .tensor import Tensor
from .terms import Binary, Funsor, Number, Unary, Variable


class FunsorProgram:
    """
    Backend program for evaluating a symbolic funsor expression.

    Programs depend on the funsor library only via ``funsor.ops`` and op
    registrations; program evaluation does not involve funsor interpretation or
    rewriting.

    :param Funsor expr: A funsor expression to evaluate.
    """

    def __init__(self, expr: Funsor):
        assert isinstance(expr, Funsor)
        self.backend = funsor.get_backend()

        # Lower and convert to A normal form.
        with funsor.interpretations.reflect:
            expr = lower(expr)
        anf = list(funsor.interpreter.anf(expr))
        ids = {}

        # Collect constants.
        constants = []
        for f in anf:
            if isinstance(f, (Number, Tensor)):
                ids[f] = len(ids)
                constants.append(f.data)
        self.constants = tuple(constants)

        # Collect input variables (the leaves).
        inputs = []
        for k, d in expr.inputs.items():
            f = Variable(k, d)
            ids[f] = len(ids)
            inputs.append(k)
        self.inputs = tuple(inputs)

        # Collect operations to be computed.
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
            else:
                raise NotImplementedError(type(f).__name__)
        self.operations = tuple(operations)

    def __call__(self, **kwargs):
        funsor.set_backend(self.backend)

        # Initialize state with constants.
        state = list(self.constants)

        # Read inputs from kwargs.
        for name in self.inputs:
            value = kwargs.pop(name, None)
            if value is None:
                raise ValueError(f"Missing kwarg: {repr(name)}")
            state.append(value)
        if kwargs:
            raise ValueError(f"Unrecognized kwargs: {set(kwargs)}")

        # Sequentially compute ops.
        for op, arg_ids in self.operations:
            args = tuple(state[i] for i in arg_ids)
            value = op(*args)
            state.append(value)

        result = state[-1]
        return result


@functools.singledispatch
def lower(x):
    raise NotImplementedError(type(x).__name__)


@lower.register(Number)
@lower.register(Tensor)
@lower.register(Variable)
def _lower_atom(x):
    return x


@lower.register(Unary)
def _lower_unary(x):
    arg = lower(x.arg)
    return Unary(x.op, arg)


@lower.register(Binary)
def _lower_binary(x):
    lhs = lower(x.lhs)
    rhs = lower(x.rhs)
    return Binary(x.op, lhs, rhs)


@lower.register(Contraction)
def _lower_contraction(x):
    if x.reduced_vars:
        raise NotImplementedError("TODO")

    terms = [lower(term) for term in x.terms]
    bin_op = functools.partial(Binary, x.bin_op)
    return functools.reduce(bin_op, terms)


__all__ = [
    "FunsorProgram",
    "lower",
]
