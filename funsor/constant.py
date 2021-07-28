# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

from funsor.delta import Delta
from funsor.distribution import Distribution
from funsor.tensor import Tensor
from funsor.terms import Binary, Funsor, Number, Unary, Variable, eager, to_data
from funsor.torch.provenance import ProvenanceTensor

from .ops import BinaryOp, FinitaryOp, GetitemOp, MatmulOp, Op, ReshapeOp, UnaryOp, AddOp


class Constant(Funsor):
    def __init__(self, const_inputs, arg):
        assert isinstance(arg, Funsor)
        assert isinstance(const_inputs, tuple)
        assert set(const_inputs).isdisjoint(arg.inputs)
        # assert all(v not in arg.inputs for v in const_inputs)
        # const_names = frozenset(v.name for v in cont_vars)
        const_inputs = OrderedDict(const_inputs)
        inputs = const_inputs.copy()
        inputs.update(arg.inputs)
        output = arg.output
        fresh = frozenset(const_inputs.keys())
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
            return Constant(tuple(const_inputs.items()), self.arg)
        return self.arg

    def eager_reduce(self, op, reduced_vars):
        assert reduced_vars.issubset(self.inputs)
        const_vars = frozenset(
            {v for v in self.const_vars if v.name not in reduced_vars}
        )
        reduced_vars = reduced_vars - frozenset({v.name for v in self.const_vars})
        if not const_vars:
            return self.arg.reduce(op, reduced_vars)
        const_inputs = tuple((v.name, v.output) for v in const_vars)
        return Constant(const_inputs, self.arg.reduce(op, reduced_vars))


@eager.register(Binary, BinaryOp, Constant, Constant)
def eager_binary_constant_constant(op, lhs, rhs):
    const_vars = lhs.const_vars | rhs.const_vars - lhs.input_vars - rhs.input_vars
    if not const_vars:
        return op(lhs.arg, rhs.arg)
    const_inputs = tuple((v.name, v.output) for v in const_vars)
    return Constant(const_inputs, op(lhs.arg, rhs.arg))


@eager.register(Binary, BinaryOp, Constant, (Number, Tensor, Distribution))
def eager_binary_constant_tensor(op, lhs, rhs):
    const_vars = lhs.const_vars - rhs.input_vars
    if not const_vars:
        return op(lhs.arg, rhs)
    const_inputs = tuple((v.name, v.output) for v in const_vars)
    return Constant(const_inputs, op(lhs.arg, rhs))


@eager.register(Binary, BinaryOp, (Number, Tensor, Distribution), Constant)
def eager_binary_tensor_constant(op, lhs, rhs):
    const_vars = rhs.const_vars - lhs.input_vars
    if not const_vars:
        return op(lhs, rhs.arg)
    const_inputs = tuple((v.name, v.output) for v in const_vars)
    return Constant(const_inputs, op(lhs, rhs.arg))


@eager.register(Unary, UnaryOp, Constant)
def eager_binary_tensor_constant(op, arg):
    const_inputs = tuple((v.name, v.output) for v in arg.const_vars)
    return Constant(const_inputs, op(arg.arg))


#  @eager.register(Binary, AddOp, Constant, Delta)
#  def eager_binary_constant_tensor(op, lhs, rhs):
#      const_vars = lhs.const_vars - rhs.input_vars
#      breakpoint()
#      if not const_vars:
#          return op(lhs.arg, rhs)
#      const_inputs = tuple((v.name, v.output) for v in const_vars)
#      return Constant(const_inputs, op(lhs.arg, rhs))
#
#
#  @eager.register(Binary, AddOp, Delta, Constant)
#  def eager_binary_tensor_constant(op, lhs, rhs):
#      const_vars = rhs.const_vars - lhs.input_vars
#      breakpoint()
#      if not const_vars:
#          return op(lhs, rhs.arg)
#      const_inputs = tuple((v.name, v.output) for v in const_vars)
#      return Constant(const_inputs, op(lhs, rhs.arg))


@to_data.register(Constant)
def constant_to_data(x, name_to_dim=None):
    data = to_data(x.arg, name_to_dim=name_to_dim)
    return ProvenanceTensor(data, provenance=frozenset((v.name, v.output) for v in x.const_vars))
