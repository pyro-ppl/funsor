# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import funsor.ops as ops
from funsor.ops import AssociativeOp, LogOp
from funsor.terms import Binary, Reduce, Tuple, Unary, eager, lazy, Variable, Number
from funsor.interpreter import interpretation
from funsor.domains import Bint, Real
from collections import defaultdict


class Tangent(Tuple):
    pass


class JVP:
    def __init__(self, primal, tangent=defaultdict(lambda: Number(0.0))):
        self.primal = primal
        self.tangent = tangent.copy()
        self.tangent[str(id(primal))] = Variable(str(id(primal)), Real)

    def __add__(self, other):
        primal = self.primal + other.primal
        tangent = defaultdict(lambda: Number(0.0))
        for key, value in self.tangent.items():
            tangent[key] += value
        for key, value in other.tangent.items():
            tangent[key] += value
        tangent[str(id(self.primal))] += other.primal - other.primal
        tangent[str(id(other.primal))] += self.primal - self.primal
        return JVP(primal, tangent)

    def __mul__(self, other):
        primal = self.primal * other.primal
        tangent = defaultdict(lambda: Number(0.0))
        for key, value in self.tangent.items():
            tangent[key] += value
        for key, value in other.tangent.items():
            tangent[key] += value
        tangent[str(id(self.primal))] *= other.primal
        tangent[str(id(other.primal))] *= self.primal
        return JVP(primal, tangent)

    def log(self):
        primal = self.primal.log()
        tangent = self.tangent
        tangent[str(id(self.primal))] /= self.primal
        return JVP(primal, tangent)



# @lazy.register(Binary, AssociativeOp, Tangent, Tangent)
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


# @lazy.register(Reduce, AssociativeOp, Tangent, frozenset)
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


@lazy.register(Unary, LogOp, Tangent)
@eager.register(Unary, LogOp, Tangent)
def jvp_log(op, arg):
    arg_primal, arg_tangent = arg
    primal = Unary(op, arg_primal)
    tangent = Binary(ops.truediv, arg_tangent, arg_primal)
    return Tangent(primal, tangent)
