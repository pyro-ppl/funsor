from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import funsor.ops as ops
from funsor.domains import find_domain
from funsor.ops import NegOp, Op
from funsor.terms import Binary, Funsor, Number, Subs, Unary, Variable, eager
from funsor.torch import Tensor


class Affine(Funsor):
    """
    Pattern representing multilinear function of input variables
    """
    def __init__(self, const, coeffs):
        assert isinstance(const, (Number, Tensor))
        assert not any(d.dtype == "real" for d in const.inputs.values())
        assert isinstance(coeffs, tuple)
        inputs = const.inputs.copy()
        output = const.output
        assert output.dtype == "real"
        for var, coeff in coeffs:
            assert isinstance(var, Variable)
            assert isinstance(coeff, (Number, Tensor))
            assert not any(d.dtype == "real" for d in coeff.inputs.values())
            inputs.update(coeff.inputs)
            inputs.update(var.inputs)
            output = find_domain(ops.add, output, find_domain(ops.mul, var.output, coeff.output))
            assert var.dtype == "real"
            assert coeff.dtype == "real"
            assert output.dtype == "real"

        super(Affine, self).__init__(inputs, output)
        self.coeffs = OrderedDict(coeffs)
        self.const = const

    def eager_subs(self, subs):
        const = Subs(self.const, subs)
        subs_dict = OrderedDict(subs)
        coeffs = tuple((var, Subs(coeff, subs))
                       for var, coeff in self.coeffs.items()
                       if var.name not in subs_dict)
        result = Affine(const, coeffs)
        for var, coeff in self.coeffs.items():
            if var.name in subs_dict:
                new_coeff = Subs(coeff, subs) * subs_dict[var.name]
                result += new_coeff
        return result


###############################################
# patterns for merging Affine with other terms
###############################################

@eager.register(Affine, (Number, Tensor), tuple)
def eager_affine(const, coeffs):
    if not coeffs:
        return const
    return None


@eager.register(Binary, Op, Affine, (Number, Tensor))
def eager_binary_affine(op, lhs, rhs):
    if op is ops.add or op is ops.sub:
        const = op(lhs.const, rhs)
        return Affine(const, tuple(lhs.coeffs.items()))
    if op is ops.mul or op is ops.truediv:
        const = op(lhs.const, rhs)
        coeffs = tuple((var, op(coeff, rhs)) for var, coeff in lhs.coeffs.items())
        return Affine(const, coeffs)
    return None


@eager.register(Binary, Op, (Number, Tensor), Affine)
def eager_binary_affine(op, lhs, rhs):
    if op is ops.add:
        const = lhs + rhs.const
        return Affine(const, tuple(rhs.coeffs.items()))
    elif op is ops.sub:
        return lhs + -rhs
    if op is ops.mul:
        const = lhs * rhs.const
        coeffs = tuple((var, lhs * coeff) for var, coeff in rhs.coeffs.items())
        return Affine(const, coeffs)
    return None


@eager.register(Binary, Op, Affine, Affine)
def eager_binary_affine_affine(op, lhs, rhs):
    if op is ops.add:
        const = lhs.const + rhs.const
        coeffs = lhs.coeffs.copy()
        for var, coeff in rhs.coeffs.items():
            if var in coeffs:
                coeffs[var] += coeff
            else:
                coeffs[var] = coeff
        return Affine(const, tuple(coeffs.items()))

    if op is ops.sub:
        return lhs + -rhs

    return None


@eager.register(Binary, Op, Affine, Variable)
def eager_binary_affine_variable(op, affine, other):
    if op is ops.add:
        const = affine.const
        coeffs = affine.coeffs.copy()
        if other in affine.inputs:
            coeffs[other] += 1
        else:
            coeffs[other] = Number(1.)
        return Affine(const, tuple(coeffs.items()))

    if op is ops.sub:
        return affine + -other

    return None


@eager.register(Binary, Op, Variable, Affine)
def eager_binary_variable_affine(op, other, affine):
    if op is ops.add:
        return affine + other

    if op is ops.sub:
        return -affine + other

    return None


@eager.register(Unary, NegOp, Affine)
def eager_negate_affine(op, affine):
    const = -affine.const
    coeffs = affine.coeffs.copy()
    for var, coeff in coeffs.items():
        coeffs[var] = -coeff
    return Affine(const, tuple(coeffs.items()))


#########################################
# patterns for creating new Affine terms
#########################################

@eager.register(Binary, Op, Variable, (Number, Tensor))
def eager_binary(op, var, other):
    if var.dtype != "real" or other.dtype != "real":
        return None

    if op is ops.add:
        const = other
        coeffs = ((var, Number(1.)),)
        return Affine(const, coeffs)
    elif op is ops.mul:
        const = Number(0.)
        coeffs = ((var, other),)
        return Affine(const, coeffs)
    elif op is ops.sub:
        return var + -other
    elif op is ops.truediv:
        return var * ops.invert(other)
    return None


@eager.register(Binary, Op, Variable, Variable)
def eager_binary(op, lhs, rhs):
    if lhs.dtype != "real" or rhs.dtype != "real":
        return None

    if op is ops.add:
        const = Number(0.)
        coeffs = ((lhs, Number(1.)), (rhs, Number(1.)))
        return Affine(const, coeffs)
    elif op is ops.sub:
        return lhs + -rhs
    return None


@eager.register(Binary, Op, (Number, Tensor), Variable)
def eager_binary(op, other, var):
    if other.dtype != "real" or var.dtype != "real":
        return None

    if op is ops.add or op is ops.mul:
        return op(var, other)
    elif op is ops.sub:
        return -var + other
    return None


@eager.register(Unary, NegOp, Variable)
def eager_negate_variable(op, var):
    if var.dtype != "real":
        return None

    const = Number(0.)
    coeffs = ((var, Number(-1, "real")),)
    return Affine(const, coeffs)
