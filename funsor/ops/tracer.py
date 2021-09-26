# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from functools import singledispatch

from .array import is_numeric_array
from .op import trace_ops
from .program import OpProgram


def trace_function(fn, kwargs: dict, *, allow_constants=False):
    """
    Traces function to an :class:`~funsor.ops.program.OpProgram` that runs on
    backend values.

    Example::

        # Create a function involving ops.
        def fn(a, x, b):
            return ops.add(ops.matmul(a, x), b)

        # Evaluate via Funsor substitution.
        data = dict(a=randn(3, 3), b=randn(3), x=randn(3))
        expected = fn(**data)

        # Alternatively evaluate via a program.
        program = trace_function(expr, data)
        actual = program(**data)
        assert (acutal == expected).all()

    :param Funsor expr: A funsor expression to evaluate.
    :returns: An op program.
    :rtype: ~funsor.ops.program.OpProgram
    """
    # Extract kwargs.
    assert isinstance(kwargs, dict)
    assert not any(is_atom(v) for v in kwargs.values())
    input_oids = {id(v) for v in kwargs.values()}
    assert len(input_oids) == len(kwargs), "repeated inputs"

    # Trace the function.
    with trace_ops() as trace:
        root = fn(**kwargs)
    assert not is_atom(root)
    trace = list(reversed(trace.values()))  # backward

    # Extract relevant portion of trace.
    dag = OrderedDict([(id(root), None)])
    for result, op, args in trace:
        if id(result) not in dag:
            continue  # not needed
        for arg in args:
            dag.setdefault(id(arg), (arg, None, None))
        dag[id(result)] = result, op, args
    anf = list(reversed(dag.values()))  # forwards

    # Collect constants (leaves).
    ids = {}
    constants = []
    for result, op, args in anf:
        if op is None and id(result) not in input_oids:
            ids[id(result)] = len(ids)
            constants.append(result)
    if constants and not allow_constants:
        raise ValueError(f"Found {len(constants)}")

    # Collect inputs (leaves).
    inputs = []
    for name, value in kwargs.items():
        ids[id(value)] = len(ids)
        inputs.append(name)

    # Collect operations to be computed (internal nodes).
    operations = []
    for result, op, args in anf:
        if id(result) in ids:
            continue  # constant or free variable
        ids[id(result)] = len(ids)
        arg_ids = tuple(ids[id(arg)] for arg in args)
        operations.append((op, arg_ids))

    return OpProgram(constants, inputs, operations)


@singledispatch
def is_atom(x):
    return is_numeric_array(x)


@is_atom.register(int)
def _is_atom_atom(x):
    return type(x) is int  # avoid numpy types


@is_atom.register(float)
def _is_atom_atom(x):
    return type(x) is float  # avoid numpy types


@is_atom.register(tuple)
def _is_atom_tuple(x):
    return all(map(is_atom, x))


__all__ = [
    "trace_function",
]
