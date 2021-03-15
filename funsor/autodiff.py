# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
import funsor.ops as ops
from funsor.ops import AssociativeOp, LogOp
from funsor.terms import Binary, Reduce, Tuple, Unary, eager, lazy, Variable, Number, Lambda, Funsor
from funsor.interpreter import interpretation
from funsor.domains import Bint, Real, Array, Reals
from collections import defaultdict
from functools import reduce, singledispatch
from funsor import Tensor
from funsor.cnf import Contraction


class JVP(Tuple):
    """
    Tuple:(Primal, Tanget)
    Semiring: (Add, Mul)
    """
    sum_op = ops.add
    prod_op = ops.mul
    div_op = ops.safediv
    zero = Number(0)
    one = Number(1)


class logJVP(Tuple):
    """
    Tuple: (LogPrimal, LogTanget)
    Semiring: (Logaddexp, Add)
    """
    sum_op = ops.logaddexp
    prod_op = ops.add
    div_op = ops.safesub
    zero = Number(-math.inf)
    one = Number(0)


def requires_grad(primal):
    tangent = Variable(str(id(primal)), Array["real", primal.data.shape])[tuple(primal.inputs)]
    return JVP(primal, tangent)


def to_var(x, name):
    var = Variable(name, Array["real", x.data.shape])[tuple(x.inputs)]
    return var


def to_arg(x):
    input_vars = tuple(Variable(key, value) for key, value in x.inputs.items())
    arg = reduce(lambda a, b: Lambda(b, a), reversed(input_vars), x)
    return arg


def fjit(cls, *args):
    new_args = []
    for arg_name, arg in zip(cls._ast_fields, args):
        if isinstance(arg, (Number, Tensor)):
            arg = to_var(arg, arg_name)
        new_args.append(arg)
    new_args = tuple(new_args)
    return cls(*new_args)


def grad(expr, targets, out_adj=None):
    out_primal, out_tangent = expr
    # in_primals = Tuple(tuple(primal for primal, _ in targets))
    in_tangents = set(tangent for _, tangent in targets)
    out_adj = Number(1) if out_adj is None else out_adj
    transposes = transpose(out_tangent, Number(1), in_tangents)
    result = {}
    for target in targets:
        result[target] = transposes[target[1]]

    #  out_shape = tuple(value.size for key, value in out_tangent.inputs.items() if key not in in_tangents.inputs)
    #  out_inputs = tuple(key for key in out_tangent.inputs if key not in in_tangents.inputs)
    #  out_tangent = Variable("dout", Array["real", out_shape])[out_inputs]
    # out_tangent = Number(1.0)
    return result


@singledispatch
def transpose(expr, out_adj, targets, result=defaultdict(lambda: Number(0))):
    breakpoint()
    if expr in targets:
        result[expr] += out_adj
        return result
    else:
        raise ValueError


@transpose.register(Contraction)
def transpose_contraction(expr, out_adj, targets, result=defaultdict(lambda: Number(0))):
    # assert expr.bin_op is ops.add or expr.bin_op is ops.logaddexp
    breakpoint()
    assert expr.red_op is ops.nullop
    if expr in targets:
        result[expr] += out_adj
        out_adj = result[expr]

    for term in expr.terms:
        if expr.bin_op is ops.add:
            term_adj = out_adj
        elif expr.bin_op is ops.mul:
            term_adj = out_adj * expr / term
        else:
            raise ValueError
        result = transpose(term, term_adj, targets, result)
    return result


@eager.register(Binary, AssociativeOp, JVP, JVP)
@eager.register(Binary, AssociativeOp, logJVP, logJVP)
def jvp_binary(op, lhs, rhs):
    sum_op, prod_op = lhs.sum_op, lhs.prod_op
    lhs_primal, lhs_tangent = lhs
    rhs_primal, rhs_tangent = rhs
    primal = Binary(op, lhs_primal, rhs_primal)
    if op is sum_op:
        tangent = sum_op(lhs_tangent, rhs_tangent)
    elif op is prod_op:
        tangent = sum_op(prod_op(rhs_primal, lhs_tangent), prod_op(lhs_primal, rhs_tangent))
    else:
        raise NotImplementedError
    return type(lhs)(primal, tangent)


@eager.register(Binary, AssociativeOp, JVP, Tensor)
@eager.register(Binary, AssociativeOp, logJVP, Tensor)
def jvp_binary_jvp_funsor(op, lhs, rhs):
    sum_op, prod_op = lhs.sum_op, lhs.prod_op
    lhs_primal, lhs_tangent = lhs
    primal = Binary(op, lhs_primal, rhs)
    if op is sum_op:
        tangent = sum_op(lhs_tangent, rhs)
    elif op is prod_op:
        tangent = prod_op(lhs_tangent, rhs)
    else:
        raise NotImplementedError
    return type(lhs)(primal, tangent)


@eager.register(Reduce, AssociativeOp, JVP, frozenset)
@eager.register(Reduce, AssociativeOp, logJVP, frozenset)
def jvp_reduce(op, arg, reduced_vars):
    sum_op, prod_op, div_op = arg.sum_op, arg.prod_op, arg.div_op
    arg_primal, arg_tangent = arg
    primal = Reduce(op, arg_primal, reduced_vars)
    if op is sum_op:
        tangent = Reduce(sum_op, arg_tangent, reduced_vars)
    elif op is prod_op:
        tangent = Reduce(prod_op, div_op(prod_op(arg_tangent, primal), arg_primal), reduced_vars)
    else:
        raise NotImplementedError
    return type(arg)(primal, tangent)


@lazy.register(Unary, LogOp, JVP)
@eager.register(Unary, LogOp, JVP)
def jvp_log(op, arg):
    arg_primal, arg_tangent = arg
    primal = Unary(op, arg_primal)
    tangent = Binary(ops.truediv, arg_tangent, arg_primal)
    return JVP(primal, tangent)
