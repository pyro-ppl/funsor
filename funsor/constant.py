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

    :class:`Constant` can be used for provenance tracking.

    Examples::

        a = Constant(OrderedDict(x=Real, y=Bint[3]), Number(0))
        a(y=1)  # returns Constant(OrderedDict(x=Real), Number(0))
        a(x=2, y=1)  # returns Number(0)

        d = Tensor(torch.tensor([1, 2, 3]))["y"]
        a + d  # returns Constant(OrderedDict(x=Real), d)

        c = Constant(OrderedDict(x=Bint[3]), Number(1))
        c.reduce(ops.add, "x")  # returns Number(3)

    :param dict const_inputs: A mapping from input name (str) to datatype (``funsor.domain.Domain``).
    :param funsor arg: A funsor that is constant wrt to const_inputs.
    """

    def __init__(self, const_inputs, arg):
        assert isinstance(arg, Funsor)
        assert isinstance(const_inputs, tuple)
        const_inputs = OrderedDict(const_inputs)
        # const_inputs and arg.inputs have to be disjoint
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
        subs = OrderedDict(subs)
        const_inputs = OrderedDict()
        for k, d in self.const_inputs.items():
            if k in subs:
                v = subs[k]
                assert v.output == d
                assert all(
                    v.inputs[k] == self.inputs[k]
                    for k in set(v.inputs).intersection(self.inputs)
                )
                const_inputs.update(
                    (name, value)
                    for name, value in v.inputs.items()
                    if name not in self.inputs
                )
            else:
                const_inputs[k] = d
        if const_inputs:
            return Constant(const_inputs, self.arg)
        return self.arg

    def eager_reduce(self, op, reduced_vars):
        assert reduced_vars.issubset(self.arg.inputs)
        return Constant(self.const_inputs, self.arg.reduce(op, reduced_vars))

    def align(self, names):
        assert isinstance(names, tuple)
        assert all(name in self.inputs for name in names)
        if not names or names == tuple(self.inputs):
            return self

        const_names = names[: len(self.const_inputs)]
        arg_names = names[len(self.const_inputs) :]
        assert frozenset(self.const_inputs) == frozenset(const_names)
        const_inputs = OrderedDict((name, self.inputs[name]) for name in const_names)
        return Constant(const_inputs, self.arg.align(arg_names))

    def materialize(self, x):
        """
        Attempt to convert a Funsor to a :class:`~funsor.terms.Number` or
        :class:`Tensor` by substituting :func:`arange` s into its free variables.

        :arg Funsor x: A funsor.
        :rtype: Funsor
        """
        assert isinstance(x, Funsor)
        if isinstance(x, (Number, Tensor)):
            return x

        assert isinstance(self.arg, Tensor)
        return self.arg.materialize(x)


@eager.register(Reduce, ops.AddOp, Constant, frozenset)
@eager.register(Reduce, ops.MulOp, Constant, frozenset)
@eager.register(Reduce, ops.LogaddexpOp, Constant, frozenset)
def eager_reduce_add(op, arg, reduced_vars):
    # reduce Constant.arg.inputs
    result = arg.arg
    if reduced_vars - arg.const_vars:
        result = result.reduce(op, reduced_vars - arg.const_vars)

    # reduce Constant.const_inputs
    reduced_const_vars = reduced_vars & arg.const_vars
    if reduced_const_vars:
        # only Bint types are supported
        assert all(var.output.dtype != "real" for var in reduced_const_vars)
        size = reduce(ops.mul, (var.output.size for var in reduced_const_vars))
        # other ops like min/max can also be supported if necessary
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
