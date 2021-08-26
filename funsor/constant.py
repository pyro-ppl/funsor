# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from functools import reduce

import funsor.ops as ops
from funsor.tensor import Tensor
from funsor.terms import (
    Binary,
    Funsor,
    FunsorMeta,
    Number,
    Reduce,
    Unary,
    Variable,
    eager,
)


class ConstantMeta(FunsorMeta):
    """
    Wrapper to convert ``const_inputs`` to a tuple.
    """

    def __call__(cls, const_inputs, arg):
        if isinstance(const_inputs, dict):
            const_inputs = tuple(const_inputs.items())

        return super(ConstantMeta, cls).__call__(const_inputs, arg)


class Constant(Funsor, metaclass=ConstantMeta):
    """
    Funsor that is constant wrt ``const_inputs``.

    ``Constant`` can be used for provenance tracking.

    Examples::

        a = Constant(OrderedDict(x=Real, y=Bint[3]), Number(0))
        assert a(y=1) is Constant(OrderedDict(x=Real), Number(0))
        assert a(x=2, y=1) is Number(0)

        d = Tensor(torch.tensor([1, 2, 3]))["y"]
        assert (a + d) is Constant(OrderedDict(x=Real), Number(0) + d)

        c = Constant(OrderedDict(x=Bint[3]), Number(1))
        assert c.reduce(ops.add, "x") is Number(3)

    :param dict const_inputs: A mapping from input name (str) to datatype (``funsor.domain.Domain``).
    :param funsor arg: A funsor that is constant wrt to const_inputs.
    """

    def __init__(self, const_inputs, arg):
        assert isinstance(arg, Funsor)
        assert isinstance(const_inputs, tuple)
        const_inputs = OrderedDict(const_inputs)
        assert set(const_inputs).isdisjoint(arg.inputs)
        inputs = const_inputs.copy()
        inputs.update(arg.inputs)
        output = arg.output
        fresh = frozenset(const_inputs)
        bound = {}
        super(Constant, self).__init__(inputs, output, fresh, bound)
        self.arg = arg
        self.const_vars = frozenset(Variable(k, v) for k, v in const_inputs.items())
        self.const_inputs = const_inputs

    def eager_subs(self, subs):
        assert isinstance(subs, tuple)
        subs = OrderedDict((k, v) for k, v in subs)
        const_inputs = OrderedDict()
        for k, d in self.const_inputs.items():
            if k in subs:
                v = subs[k]
                const_inputs.update(
                    (name, value)
                    for name, value in v.inputs.items()
                    if name not in self.arg.inputs
                )
            else:
                const_inputs[k] = d
        if const_inputs:
            return Constant(const_inputs, self.arg)
        return self.arg

    def eager_reduce(self, op, reduced_vars):
        assert reduced_vars.issubset(self.arg.inputs)
        return Constant(self.const_inputs, self.arg.reduce(op, reduced_vars))


@eager.register(Reduce, (ops.AddOp, ops.MulOp, ops.LogaddexpOp), Constant, frozenset)
def eager_reduce_add(op, arg, reduced_vars):
    # reduce Constant.arg.inputs
    result = arg.arg
    if reduced_vars - arg.const_vars:
        result = result.reduce(op, reduced_vars - arg.const_vars)

    # reduce Constant.const_inputs
    reduced_const_vars = reduced_vars & arg.const_vars
    if reduced_const_vars:
        assert all(var.output.dtype != "real" for var in reduced_const_vars)
        size = reduce(ops.mul, (var.output.size for var in reduced_const_vars))
        if op is ops.add:
            prod_op = ops.mul
        elif op is ops.mul:
            prod_op = ops.pow
        elif op is ops.logaddexp:
            prod_op = ops.add
            size = ops.log(size)
        result = prod_op(result, size)
        const_vars = arg.const_vars - reduced_const_vars
        const_inputs = OrderedDict((v.name, v.output) for v in const_vars)
        if const_inputs:
            return Constant(const_inputs, result)
        return result
    return Constant(arg.const_inputs, result)


@eager.register(Binary, ops.BinaryOp, Constant, Constant)
def eager_binary_constant_constant(op, lhs, rhs):
    const_vars = (
        (lhs.const_vars | rhs.const_vars) - lhs.arg.input_vars - rhs.arg.input_vars
    )
    const_inputs = OrderedDict((v.name, v.output) for v in const_vars)
    if const_inputs:
        return Constant(const_inputs, op(lhs.arg, rhs.arg))
    return op(lhs.arg, rhs.arg)


@eager.register(Binary, ops.BinaryOp, Constant, (Number, Tensor))
def eager_binary_constant_tensor(op, lhs, rhs):
    const_inputs = OrderedDict(
        (k, v) for k, v in lhs.const_inputs.items() if k not in rhs.inputs
    )
    if const_inputs:
        return Constant(const_inputs, op(lhs.arg, rhs))
    return op(lhs.arg, rhs)


@eager.register(Binary, ops.BinaryOp, (Number, Tensor), Constant)
def eager_binary_tensor_constant(op, lhs, rhs):
    const_inputs = OrderedDict(
        (k, v) for k, v in rhs.const_inputs.items() if k not in lhs.inputs
    )
    if const_inputs:
        return Constant(const_inputs, op(lhs, rhs.arg))
    return op(lhs, rhs.arg)


@eager.register(Unary, ops.UnaryOp, Constant)
def eager_unary(op, arg):
    return Constant(arg.const_inputs, op(arg.arg))
