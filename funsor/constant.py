# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

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
    to_funsor,
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
    """
    Constant funsor wrt to multiple variables (``const_inputs``).

    This can be used for provenance tracking.

    ``const_inputs`` are ignored (removed) under
    substition/reduction/binary operations::

        a = Constant(OrderedDict(x=Real, y=Bint[3]), arg)
        assert a.reduce(ops.add, "x") is Constant(OrderedDict(y=Bint[3]), arg)
        assert a(y=1) is Constant(OrderedDict(x=Real), arg)
        assert a(x=0, y=1) is arg

        c = Normal(0, 1, value="x")
        assert (a + c) is Constant(OrderedDict(y=Bint[3]), arg + c)

        d = Tensor(torch.tensor([1, 2, 3]))["y"]
        assert (a + d) is Constant(OrderedDict(x=Real), arg + d)

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
            # handle when subs is in self.arg.inputs
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
    const_vars = (
        (lhs.const_vars | rhs.const_vars) - lhs.arg.input_vars - rhs.arg.input_vars
    )
    const_inputs = OrderedDict((v.name, v.output) for v in const_vars)
    if const_inputs:
        return Constant(const_inputs, op(lhs.arg, rhs.arg))
    return op(lhs.arg, rhs.arg)


@eager.register(Binary, BinaryOp, Constant, (Number, Tensor))
def eager_binary_constant_tensor(op, lhs, rhs):
    const_inputs = OrderedDict(
        (k, v) for k, v in lhs.const_inputs.items() if k not in rhs.inputs
    )
    if const_inputs:
        return Constant(const_inputs, op(lhs.arg, rhs))
    return op(lhs.arg, rhs)


@eager.register(Binary, BinaryOp, (Number, Tensor), Constant)
def eager_binary_tensor_constant(op, lhs, rhs):
    const_inputs = OrderedDict(
        (k, v) for k, v in rhs.const_inputs.items() if k not in lhs.inputs
    )
    if const_inputs:
        return Constant(const_inputs, op(lhs, rhs.arg))
    return op(lhs, rhs.arg)


@eager.register(Unary, UnaryOp, Constant)
def eager_unary(op, arg):
    return Constant(arg.const_inputs, op(arg.arg))


@to_data.register(Constant)
def constant_to_data(x, name_to_dim=None):
    data = to_data(x.arg, name_to_dim=name_to_dim)
    return ProvenanceTensor(data, provenance=frozenset(x.const_inputs.items()))


@to_funsor.register(ProvenanceTensor)
def provenance_to_funsor(x, output=None, dim_to_name=None):
    if isinstance(x, ProvenanceTensor):
        ret = to_funsor(x._t, output=output, dim_to_name=dim_to_name)
        return Constant(OrderedDict(x._provenance), ret)
