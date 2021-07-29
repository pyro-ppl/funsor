# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

from funsor.distribution import Distribution
from funsor.tensor import Tensor
from funsor.terms import (
    Binary,
    Funsor,
    FunsorMeta,
    Number,
    Unary,
    Variable,
    eager,
    to_data,
)
from funsor.torch.provenance import ProvenanceTensor

from .ops import BinaryOp, UnaryOp


class ConstantMeta(FunsorMeta):
    """
    Wrapper to convert ``const_inputs`` to a tuple.
    """

    def __call__(cls, const_inputs, arg):
        if isinstance(const_inputs, dict):
            const_inputs = tuple(const_inputs.items())

        return super(ConstantMeta, cls).__call__(const_inputs, arg)


class Constant(Funsor, metaclass=ConstantMeta):
    def __init__(self, const_inputs, arg):
        assert isinstance(arg, Funsor)
        assert isinstance(const_inputs, tuple)
        assert set(const_inputs).isdisjoint(arg.inputs)
        const_inputs = OrderedDict(const_inputs)
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
            # handle when subs is in self.arg.inputs
            if k in subs:
                v = subs[k]
                if isinstance(v, Variable):
                    del subs[k]
                    k = v.name
                    const_inputs[k] = d
        if const_inputs:
            return Constant(const_inputs, self.arg)
        return self.arg

    def eager_reduce(self, op, reduced_vars):
        assert reduced_vars.issubset(self.inputs)
        const_inputs = OrderedDict(
            (k, v) for k, v in self.const_inputs.items() if k not in reduced_vars
        )
        reduced_vars = reduced_vars - frozenset(self.const_inputs)
        reduced_arg = self.arg.reduce(op, reduced_vars)
        if const_inputs:
            return Constant(const_inputs, reduced_arg)
        return reduced_arg


@eager.register(Binary, BinaryOp, Constant, Constant)
def eager_binary_constant_constant(op, lhs, rhs):
    const_inputs = OrderedDict(
        (k, v) for k, v in lhs.const_inputs.items() if k not in rhs.const_inputs
    )
    const_inputs.update(
        (k, v) for k, v in rhs.const_inputs.items() if k not in lhs.const_inputs
    )
    if const_inputs:
        return Constant(const_inputs, op(lhs.arg, rhs.arg))
    return op(lhs.arg, rhs.arg)


@eager.register(Binary, BinaryOp, Constant, (Number, Tensor, Distribution))
def eager_binary_constant_tensor(op, lhs, rhs):
    const_inputs = OrderedDict(
        (k, v) for k, v in lhs.const_inputs.items() if k not in rhs.inputs
    )
    if const_inputs:
        return Constant(const_inputs, op(lhs.arg, rhs))
    return op(lhs.arg, rhs)


@eager.register(Binary, BinaryOp, (Number, Tensor, Distribution), Constant)
def eager_binary_tensor_constant(op, lhs, rhs):
    const_inputs = OrderedDict(
        (k, v) for k, v in rhs.const_inputs.items() if k not in lhs.inputs
    )
    if const_inputs:
        return Constant(const_inputs, op(lhs, rhs.arg))
    return op(lhs, rhs.arg)


@eager.register(Unary, UnaryOp, Constant)
def eager_binary_tensor_constant(op, arg):
    return Constant(arg.const_inputs, op(arg.arg))


@to_data.register(Constant)
def constant_to_data(x, name_to_dim=None):
    data = to_data(x.arg, name_to_dim=name_to_dim)
    return ProvenanceTensor(data, provenance=frozenset(x.const_inputs.items()))
