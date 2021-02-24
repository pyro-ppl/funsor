# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import funsor.ops as ops
from funsor.ops import AssociativeOp, LogOp
from funsor.terms import Binary, Reduce, Tuple, Unary, eager


class Tangent(Tuple):
    pass


@eager.register(Binary, AssociativeOp, Tangent, Tangent)
def jvp_binary(op, lhs, rhs):
    lhs_primal, lhs_tangent = lhs
    rhs_primal, rhs_tangent = rhs
    primal = Binary(op, lhs_primal, rhs_primal)
    if op is ops.add:
        tangent = lhs_tangent + rhs_tangent
    elif op is ops.mul:
        tangent = rhs_primal * lhs_tangent + lhs_primal * rhs_tangent
    else:
        raise NotImplementedError
    return Tangent(primal, tangent)


@eager.register(Reduce, AssociativeOp, Tangent, frozenset)
def jvp_reduce(op, arg, reduced_vars):
    arg_primal, arg_tangent = arg
    primal = Reduce(op, arg_primal, reduced_vars)
    if op is ops.add:
        tangent = Reduce(op, arg_tangent, reduced_vars)
    elif op is ops.mul:
        tangent = Reduce(ops.add, arg_tangent * primal / arg_primal, reduced_vars)
    else:
        raise NotImplementedError
    return Tangent(primal, tangent)


@eager.register(Unary, LogOp, Tangent)
def jvp_log(op, arg):
    arg_primal, arg_tangent = arg
    primal = Unary(op, arg_primal)
    tangent = Binary(ops.truediv, arg_tangent, arg_primal)
    return Tangent(primal, tangent)
