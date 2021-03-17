# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
from collections import defaultdict
from functools import reduce, singledispatch

import funsor.ops as ops
from funsor import Tensor
from funsor.adjoint import _alpha_unmangle
from funsor.cnf import Contraction
from funsor.domains import Array, Bint, Real, Reals
from funsor.interpretations import autodiff, trace
from funsor.interpreter import interpretation
from funsor.ops import AssociativeOp, LogOp
from funsor.terms import (
    Binary,
    Funsor,
    Lambda,
    Number,
    Reduce,
    Tuple,
    Unary,
    Variable,
    eager,
    lazy,
)


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

    @property
    def primal(self):
        return self[0]

    @property
    def tangent(self):
        return self[1]


class LJVP(Tuple):
    """
    Tuple: (LogPrimal, LogTanget)
    Semiring: (Logaddexp, Add)
    """

    sum_op = ops.logaddexp
    prod_op = ops.add
    div_op = ops.safesub
    zero = Number(-math.inf)
    one = Number(0)

    @property
    def primal(self):
        return self[0]

    @property
    def tangent(self):
        return self[1]


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


def to_jvp(primal):
    input_vars = tuple(Variable(key, value) for key, value in primal.inputs.items())
    output = reduce(lambda x, y: Lambda(y, x), reversed(input_vars), primal).output
    tangent_placeholder = Variable(str(id(primal)), output)[tuple(primal.inputs)]
    return JVP(primal, tangent_placeholder)


def to_ljvp(primal):
    input_vars = tuple(Variable(key, value) for key, value in primal.inputs.items())
    output = reduce(lambda x, y: Lambda(y, x), reversed(input_vars), primal).output
    tangent_placeholder = Variable(str(id(primal)), output)[tuple(primal.inputs)]
    return LJVP(primal, tangent_placeholder)


def grad(expr, targets, out_tangent=None):
    out_tangent = expr.one if out_tangent is None else out_tangent
    in_tangents = set(target.tangent for target in targets)
    transposes = transpose(
        expr.tangent, out_tangent, in_tangents, defaultdict(lambda: expr.zero)
    )
    result = {}
    for target in targets:
        result[target] = transposes[target.tangent]
    return result


@singledispatch
def transpose(expr, out_tangent, in_tangents, result):
    if expr in in_tangents:
        result[expr] += out_tangent
    return result


@transpose.register(Binary)
def transpose_binary(expr, out_tangent, in_tangents, result):
    if expr in in_tangents:
        result[expr] += out_tangent
        out_tangent = result[expr]

    op, lhs, rhs = expr.op, expr.lhs, expr.rhs

    if op is ops.add:
        lhs_adj = out_tangent.reduce(ops.add, out_tangent.input_vars - lhs.input_vars)
        rhs_adj = out_tangent.reduce(ops.add, out_tangent.input_vars - rhs.input_vars)
    elif op is ops.mul:
        lhs_adj = (out_tangent * rhs).reduce(
            ops.add, out_tangent.input_vars - lhs.input_vars
        )
        rhs_adj = (out_tangent * lhs).reduce(
            ops.add, out_tangent.input_vars - rhs.input_vars
        )
    else:
        return result  # is it always correct?
    result = transpose(lhs, lhs_adj, in_tangents, result)
    result = transpose(rhs, rhs_adj, in_tangents, result)
    return result


@transpose.register(Reduce)
def transpose_reduce(expr, out_tangent, in_tangents, result):
    if expr in in_tangents:
        result[expr] += out_tangent
        out_tangent = result[expr]

    # fix this in contraction as well
    op, arg, reduced_vars = _alpha_unmangle(expr)

    if op is ops.add:
        arg_adj = out_tangent.expand(ops.add, tuple(reduced_vars))
    elif op is ops.mul:
        arg_adj = ops.safediv(ops.mul(out_tangent, expr), arg)
    else:
        raise ValueError
    result = transpose(arg, arg_adj, in_tangents, result)
    return result


@transpose.register(Contraction)
def transpose_contraction(expr, out_tangent, in_tangents, result):
    breakpoint()
    if expr in in_tangents:
        result[expr] += out_tangent
        out_tangent = result[expr]

    if expr.red_op is ops.nullop:
        for term in expr.terms:
            if expr.bin_op is ops.add:
                term_adj = out_tangent.reduce(
                    ops.add, out_tangent.input_vars - term.input_vars
                )
            elif expr.bin_op is ops.mul:
                expr_div_term = reduce(
                    ops.mul, tuple(t for t in expr.terms if t is not term)
                )
                term_adj = (out_tangent * expr_div_term).reduce(
                    ops.add, out_tangent.input_vars - term.input_vars
                )
            else:
                raise ValueError
            result = transpose(term, term_adj, in_tangents, result)
    elif expr.bin_op is ops.nullop:
        for term in expr.terms:  # only one term
            if expr.red_op is ops.add:
                term_adj = out_tangent.expand(ops.add, tuple(expr.reduced_vars))
            elif expr.red_op is ops.mul:
                term_adj = ops.safediv(ops.mul(out_tangent, expr), term)
            else:
                raise ValueError
            result = transpose(term, term_adj, in_tangents, result)
    else:
        raise ValueError
    return result


@eager.register(Binary, AssociativeOp, JVP, JVP)
@eager.register(Binary, AssociativeOp, LJVP, LJVP)
@autodiff.register(Binary, AssociativeOp, JVP, JVP)
@autodiff.register(Binary, AssociativeOp, LJVP, LJVP)
def jvp_binary(op, lhs, rhs):
    sum_op, prod_op = lhs.sum_op, lhs.prod_op
    lhs_primal, lhs_tangent = lhs
    rhs_primal, rhs_tangent = rhs
    primal = Binary(op, lhs_primal, rhs_primal)
    if op is sum_op:
        tangent = sum_op(lhs_tangent, rhs_tangent)
    elif op is prod_op:
        tangent = sum_op(
            prod_op(rhs_primal, lhs_tangent), prod_op(lhs_primal, rhs_tangent)
        )
    else:
        raise NotImplementedError
    return type(lhs)(primal, tangent)


@eager.register(Binary, AssociativeOp, JVP, Tensor)
@eager.register(Binary, AssociativeOp, LJVP, Tensor)
@autodiff.register(Binary, AssociativeOp, JVP, Tensor)
@autodiff.register(Binary, AssociativeOp, LJVP, Tensor)
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
@eager.register(Reduce, AssociativeOp, LJVP, frozenset)
@autodiff.register(Reduce, AssociativeOp, JVP, frozenset)
@autodiff.register(Reduce, AssociativeOp, LJVP, frozenset)
def jvp_reduce(op, arg, reduced_vars):
    sum_op, prod_op, div_op = arg.sum_op, arg.prod_op, arg.div_op
    arg_primal, arg_tangent = arg
    out_primal = Reduce(op, arg_primal, reduced_vars)
    if op is sum_op:
        tangent = Reduce(sum_op, arg_tangent, reduced_vars)
    elif op is prod_op:
        tangent = Reduce(
            prod_op, div_op(prod_op(arg_tangent, out_primal), arg_primal), reduced_vars
        )
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
