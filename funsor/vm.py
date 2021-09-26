# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from .interpreter import anf
from .terms import Binary, Funsor, Unary, Variable


class FunsorProgram:
    """
    Programs should be pickleable, but be sure to ``funsor.set_backend(...)``
    before running an unpickled program.

    :param Funsor expr:
    """

    def __init__(self, expr: Funsor):
        assert isinstance(expr, Funsor)

        # Collect input variables (the leaves).
        self.args = tuple(expr.inputs)
        ids = {}
        for k, d in expr.inputs.items():
            ids[Variable(k, d)] = len(ids)

        # Traverse expression.
        operations = []
        for f in anf(expr):
            if f in ids:
                continue
            ids[f] = len(ids)
            if isinstance(f, Unary):
                input_ids = (ids[f.arg],)
                operations.append((f.op, input_ids))
            elif isinstance(f, Binary):
                input_ids = (ids[f.lhs], ids[f.rhs])
                operations.append((f.op, input_ids))
            else:
                raise NotImplementedError(type(f).__name__)

        self.operations = tuple(operations)

    def __call__(self, **kwargs):
        # Initialize state with kwargs.
        state = []
        for arg in self.args:
            state.append(kwargs.pop(arg))
        assert not kwargs

        # Sequentially apply each op.
        for op, input_ids in self.operations:
            values = tuple(state[i] for i in input_ids)
            value = op(*values)
            state.append(value)

        return state[-1]
