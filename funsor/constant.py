# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

from funsor.delta import Delta
from funsor.distribution import Distribution
from funsor.tensor import Tensor
from funsor.terms import Binary, Funsor, Number, Unary, Variable, eager, to_funsor
from funsor.torch import MetadataTensor

from .ops import BinaryOp, FinitaryOp, GetitemOp, MatmulOp, Op, ReshapeOp, UnaryOp


class Constant(Funsor):
    def __init__(self, const_vars, arg):
        assert isinstance(arg, Funsor)
        assert isinstance(const_vars, frozenset)
        assert all(isinstance(v, Variable) for v in const_vars)
        assert all(v not in arg.inputs for v in const_vars)
        # const_names = frozenset(v.name for v in cont_vars)
        inputs = OrderedDict((v.name, v.output) for v in const_vars)
        inputs.update(arg.inputs)
        output = arg.output
        fresh = const_vars
        bound = {}
        super(Constant, self).__init__(inputs, output, fresh, bound)
        self.arg = arg
        self.const_vars = const_vars

    def eager_subs(self, subs):
        assert isinstance(subs, tuple)
        const_vars = self.const_vars
        for name, value in subs:
            if isinstance(value, Variable):
                breakpoint()
                continue

            breakpoint()
            if isinstance(value, (Number, Tensor)):
                const_vars = const_vars - value

        return Constant(const_vars, self.arg)

    def eager_reduce(self, op, reduced_vars):
        assert reduced_vars.issubset(self.inputs)
        const_vars = frozenset(
            {v for v in self.const_vars if v.name not in reduced_vars}
        )
        reduced_vars = reduced_vars - frozenset({v.name for v in self.const_vars})
        if not const_vars:
            return self.arg.reduce(op, reduced_vars)
        return Constant(const_vars, self.arg.reduce(op, reduced_vars))


@eager.register(Binary, BinaryOp, Constant, Constant)
def eager_binary_constant_constant(op, lhs, rhs):
    const_vars = lhs.const_vars | rhs.const_vars - lhs.input_vars - rhs.input_vars
    if not const_vars:
        return op(lhs.arg, rhs.arg)
    return Constant(const_vars, op(lhs.arg, rhs.arg))


@eager.register(Binary, BinaryOp, Constant, (Number, Tensor))
def eager_binary_constant_tensor(op, lhs, rhs):
    const_vars = lhs.const_vars - rhs.input_vars
    if not const_vars:
        return op(lhs.arg, rhs)
    return Constant(const_vars, op(lhs.arg, rhs))


@eager.register(Binary, BinaryOp, (Number, Tensor), Constant)
def eager_binary_tensor_constant(op, lhs, rhs):
    const_vars = rhs.const_vars - lhs.input_vars
    if not const_vars:
        return op(lhs, rhs.arg)
    return Constant(const_vars, op(lhs, rhs.arg))


@eager.register(Unary, UnaryOp, Constant)
def eager_binary_tensor_constant(op, arg):
    return Constant(arg.const_vars, op(arg.arg))


@to_funsor.register(MetadataTensor)
def tensor_to_funsor(x, output=None, dim_to_name=None):
    breakpoint()
    if not dim_to_name:
        output = output if output is not None else Reals[x.shape]
        result = Tensor(x, dtype=output.dtype)
        if result.output != output:
            raise ValueError(
                "Invalid shape: expected {}, actual {}".format(
                    output.shape, result.output.shape
                )
            )
        return result
    else:
        assert all(
            isinstance(k, int) and k < 0 and isinstance(v, str)
            for k, v in dim_to_name.items()
        )

        if output is None:
            # Assume the leftmost dim_to_name key refers to the leftmost dim of x
            # when there is ambiguity about event shape
            batch_ndims = min(-min(dim_to_name.keys()), len(x.shape))
            output = Reals[x.shape[batch_ndims:]]

        # logic very similar to pyro.ops.packed.pack
        # this should not touch memory, only reshape
        # pack the tensor according to the dim => name mapping in inputs
        packed_inputs = OrderedDict()
        for dim, size in zip(range(len(x.shape) - len(output.shape)), x.shape):
            name = dim_to_name.get(dim + len(output.shape) - len(x.shape), None)
            if name is not None and size != 1:
                packed_inputs[name] = Bint[size]
        shape = tuple(d.size for d in packed_inputs.values()) + output.shape
        if x.shape != shape:
            x = x.reshape(shape)
        return Tensor(x, packed_inputs, dtype=output.dtype)
