from collections import OrderedDict

from funsor.terms import Funsor, Variable, Binary, eager, Number, Unary
from .ops import BinaryOp, FinitaryOp, GetitemOp, MatmulOp, Op, ReshapeOp, UnaryOp
from funsor.tensor import Tensor
from funsor.delta import Delta
from funsor.distribution import Distribution

class Constant(Funsor):
    def __init__(self, const_vars, arg):
        assert isinstance(arg, Funsor)
        assert isinstance(const_vars, frozenset)
        assert all(isinstance(v, Variable) for v in const_vars)
        assert all(v not in arg.inputs for v in const_vars)
        # const_names = frozenset(v.name for v in cont_vars)
        inputs = OrderedDict(
            (v.name, v.output) for v in const_vars
        )
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
        const_vars = frozenset({v for v in self.const_vars if v.name not in reduced_vars})
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
