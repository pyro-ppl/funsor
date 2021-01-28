# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from functools import singledispatch

import funsor.ops as ops
from funsor.cnf import Contraction
from funsor.terms import Binary, Number, Reduce


@singledispatch
def transpose(expr):
    result = defaultdict(lambda: Number(0))
    result[expr] = expr
    return result


@transpose.register(Binary)
def _(expr):
    lhs_adj = transpose(expr.lhs)
    rhs_adj = transpose(expr.rhs)
    result = defaultdict(lambda: Number(0))
    if expr.op is ops.add:
        for key, value in rhs_adj.items():
            result[key] += value
        for key, value in lhs_adj.items():
            result[key] += value
        return result
    if expr.op is ops.mul:
        for key, value in rhs_adj.items():
            result[key] += expr.lhs * value
        for key, value in lhs_adj.items():
            result[key] += value * expr.rhs
        return result
    raise NotImplementedError


@transpose.register(Reduce)
def _(expr):
    arg_adj = transpose(expr.arg)
    if expr.op is ops.add:
        return arg_adj
    if expr.op is ops.mul:
        result = defaultdict(lambda: Number(0))
        for k, v in arg_adj.values():
            result[k] = ops.safediv(
                v * expr.arg, expr.arg.reduce(ops.mul, expr.reduced_vars))
        return result
    raise NotImplementedError


@transpose.register(Contraction)
def _(expr):
    if expr.bin_op is ops.nullop and expr.red_op is ops.add:
        assert len(expr.terms) == 1
        term, = expr.terms
        return transpose(term)
    if expr.bin_op is ops.add and expr.red_op is ops.nullop:
        result = defaultdict(lambda: Number(0))
        for term in expr.terms:
            for key, value in transpose(term).items():
                result[key] += value
        return result
    if expr.bin_op is ops.mul:
        assert len(expr.terms) == 2
        lhs, rhs = expr.terms
        lhs_adj = transpose(lhs)
        rhs_adj = transpose(rhs)
        result = defaultdict(lambda: Number(0))
        for key, value in lhs_adj.items():
            result[key] += value * rhs
        for key, value in rhs_adj.items():
            result[key] += lhs * value
        return result
    raise NotImplementedError
