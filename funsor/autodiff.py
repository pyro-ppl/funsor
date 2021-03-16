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
from funsor.interpretations import trace, autodiff


@trace.register(Binary, AssociativeOp, Funsor, Funsor)
def trace_binary_associativeop(op, lhs, rhs):
    with lazy:
        result = Binary(op, lhs, rhs)
    return result


@trace.register(Reduce, AssociativeOp, Funsor, frozenset)
def trace_binary_associativeop(op, arg, reduced_args):
    with lazy:
        result = Reduce(op, arg, reduced_args)
    return result


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
    transposes = transpose(out_tangent, out_adj, in_tangents)
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


@transpose.register(Binary)
def transpose_binary(expr, out_adj, targets, result=defaultdict(lambda: Number(0))):
    breakpoint()
    if expr in targets:
        result[expr] += out_adj
        out_adj = result[expr]

    lhs, rhs, op = expr.lhs, expr.rhs, expr.op

    if op is ops.add:
        lhs_adj = out_adj.reduce(ops.add, out_adj.input_vars - lhs.input_vars)
        rhs_adj = out_adj.reduce(ops.add, out_adj.input_vars - rhs.input_vars)
    elif op is ops.mul:
        lhs_adj = (out_adj * rhs).reduce(ops.add, out_adj.input_vars - lhs.input_vars)
        rhs_adj = (out_adj * lhs).reduce(ops.add, out_adj.input_vars - rhs.input_vars)
    else:
        return result # is it always correct?
    result = transpose(lhs, lhs_adj, targets, result)
    result = transpose(rhs, rhs_adj, targets, result)
    return result


@transpose.register(Reduce)
def transpose_reduce(expr, out_adj, targets, result=defaultdict(lambda: Number(0))):
    breakpoint()
    if expr in targets:
        result[expr] += out_adj
        out_adj = result[expr]

    op, arg, reduced_vars = expr.op, expr.arg, expr.reduced_vars

    if op is ops.add:
        arg_adj = out_adj.expand(ops.add, tuple(reduced_vars))
    elif op is ops.mul:
        arg_adj = ops.safediv(ops.mul(out_adj, expr), arg)
    else:
        raise ValueError
    result = transpose(arg, arg_adj, targets, result)
    return result


@transpose.register(Contraction)
def transpose_contraction(expr, out_adj, targets, result=defaultdict(lambda: Number(0))):
    # assert expr.bin_op is ops.add or expr.bin_op is ops.logaddexp
    breakpoint()
    if expr in targets:
        result[expr] += out_adj
        out_adj = result[expr]

    if expr.red_op is ops.nullop:
        for term in expr.terms:
            if expr.bin_op is ops.add:
                term_adj = out_adj.reduce(ops.add, out_adj.input_vars - term.input_vars)
            elif expr.bin_op is ops.mul:
                expr_div_term = reduce(ops.mul, tuple(t for t in expr.terms if t is not term))
                term_adj = (out_adj * expr_div_term).reduce(ops.add, out_adj.input_vars - term.input_vars)
            else:
                raise ValueError
            result = transpose(term, term_adj, targets, result)
    elif expr.bin_op is ops.nullop:
        for term in expr.terms: # only one term
            if expr.red_op is ops.add:
                term_adj = out_adj.expand(ops.add, tuple(expr.reduced_vars))
            elif expr.red_op is ops.mul:
                term_adj = ops.safediv(ops.mul(out_adj, expr), term)
            else:
                raise ValueError
            result = transpose(term, term_adj, targets, result)
    else:
        raise ValueError
    return result


@autodiff.register(Binary, AssociativeOp, JVP, JVP)
@autodiff.register(Binary, AssociativeOp, logJVP, logJVP)
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


@autodiff.register(Binary, AssociativeOp, JVP, Tensor)
@autodiff.register(Binary, AssociativeOp, logJVP, Tensor)
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


@autodiff.register(Reduce, AssociativeOp, JVP, frozenset)
@autodiff.register(Reduce, AssociativeOp, logJVP, frozenset)
def jvp_reduce(op, arg, reduced_vars):
    sum_op, prod_op, div_op = arg.sum_op, arg.prod_op, arg.div_op
    arg_primal, arg_tangent = arg
    out_primal = Reduce(op, arg_primal, reduced_vars)
    if op is sum_op:
        tangent = Reduce(sum_op, arg_tangent, reduced_vars)
    elif op is prod_op:
        tangent = Reduce(prod_op, div_op(prod_op(arg_tangent, out_primal), arg_primal), reduced_vars)
    else:
        raise NotImplementedError
    return type(arg)(out_primal, tangent)


#  @lazy.register(Unary, LogOp, JVP)
#  @eager.register(Unary, LogOp, JVP)
#  def jvp_log(op, arg):
#      arg_primal, arg_tangent = arg
#      primal = Unary(op, arg_primal)
#      tangent = Binary(ops.truediv, arg_tangent, arg_primal)
#      return JVP(primal, tangent)
