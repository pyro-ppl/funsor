# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import inspect
import typing

from funsor import interpreter
from funsor.domains import BintType, Dependent, find_domain
from funsor.interpretations import eager
from funsor.interpreter import PatternMissingError
from funsor.tensor import Tensor
from funsor.terms import Binary, Number, Tuple, Unary, to_data, to_funsor
from funsor.typing import deep_issubclass
from funsor.util import as_callable

from . import ops


def eager_tensor_made_op(op, *args):
    name_to_dim = {}
    for arg in args:
        for k, v in reversed(arg.inputs.items()):
            if isinstance(v, BintType):
                name_to_dim.setdefault(k, -1 - len(name_to_dim))
    dim_to_name = {dim: name for name, dim in name_to_dim.items()}
    try:
        raw_args = [to_data(arg, name_to_dim=name_to_dim) for arg in args]
    except PatternMissingError:
        return None  # give up
    data = op(*raw_args)
    output = find_domain(op, *(arg.output for arg in args))
    return to_funsor(data, output, dim_to_name=dim_to_name)


def make_op(fn):
    """
    Wrap a python function for use in Funsor.

    1. Creates a new ``Op`` subclass and instance ``op``.
    2. Registers a :func:`~funsor.domains.find_domain` rule based on ``fn``'s
       return type.
    3. Registers an eager rule.

    This assumes ``fn`` is compatible with broadcasting.
    """
    input_types = typing.get_type_hints(as_callable(fn))
    output_type = input_types.pop("return")
    parameters = tuple(inspect.Signature.from_callable(as_callable(fn)).parameters)
    hints = tuple(input_types.get(name) for name in parameters)
    arity = len(parameters)
    if arity == 1:
        op_cls = ops.UnaryOp
        funsor_cls = Unary
    elif arity == 2:
        op_cls = ops.BinaryOp
        funsor_cls = Binary
    else:
        raise NotImplementedError("TODO convert to a finitary")
    op = op_cls.make(fn)

    # Register a find_domain implementation.
    @find_domain.register(type(op))
    def find_domain_made_op(op, *args):
        if interpreter._TYPECHECK:
            for arg, hint in zip(args, hints):
                if hint is not None:
                    assert deep_issubclass(arg, hint)
        if isinstance(output_type, Dependent):
            return output_type(**dict(zip(parameters, args)))
        return output_type

    # Register an eager funsor rule.
    # TODO generalize to more funsor types, ideally to Funsor itself.
    pattern = [funsor_cls, type(op)] + [(Number, Tuple, Tensor)] * arity
    eager.register(*pattern)(eager_tensor_made_op)

    return op
