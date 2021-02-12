# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import math
from collections import defaultdict
from functools import singledispatch

import funsor.ops as ops
from funsor.cnf import Contraction
from funsor.terms import Binary, Number, Reduce


class Semiring:
    @staticmethod
    def backward_reduce(expr):
        return transpose(expr.arg)


class SumProduct(Semiring):
    sum_op = ops.add
    prod_op = ops.mul
    div_op = ops.safediv
    zero = Number(0)
    one = Number(1)


class Tropical(Semiring):
    sum_op = ops.max
    prod_op = ops.add
    div_op = ops.safesub
    zero = Number(-math.inf)
    one = Number(0)

    def backward_reduce(self, expr):
        # TODO these should be a funsor pattern
        raise NotImplementedError


SEMIRING = SumProduct


@contextlib.contextmanager
def set_semiring(new):
    assert isinstance(new, Semiring)
    global SEMIRING
    old = SEMIRING
    try:
        SEMIRING = new
        yield
    finally:
        SEMIRING = old


@singledispatch
def transpose(expr):
    """
    Similar to adjoint, but currently hard-coded to sum-product semiring.
    """
    result = defaultdict(lambda: SEMIRING.zero)
    result[expr] = expr
    return result


@transpose.register(Binary)
def _(expr):
    lhs_adj = transpose(expr.lhs)
    rhs_adj = transpose(expr.rhs)
    result = defaultdict(lambda: SEMIRING.zero)
    if expr.op is SEMIRING.sum_op:
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
    if isinstance(expr.op, ops.GetitemOp):
        for key, value in transpose(expr.lhs).items():
            result[key] += expr.op(value, expr.rhs)
        return result
    raise NotImplementedError


@transpose.register(Reduce)
def _(expr):
    if expr.op is ops.add:
        # Version 0.
        return transpose(expr.arg)
        # Version 1.
        # return SEMIRING.backward_reduce(expr)
        # Version 2.
        # return Argreduce(SEMIRING, expr.arg, expr.reduced_vars)
    if expr.op is ops.mul:
        arg_adj = transpose(expr.arg)
        result = defaultdict(lambda: SEMIRING.zero)
        for k, v in arg_adj.values():
            result[k] = ops.safediv(
                v * expr.arg, expr.arg.reduce(ops.mul, expr.reduced_vars)
            )
        return result
    raise NotImplementedError


@transpose.register(Contraction)
def _(expr):
    if expr.bin_op is ops.nullop and expr.red_op is ops.add:
        assert len(expr.terms) == 1
        (term,) = expr.terms
        return transpose(term)
    if expr.bin_op is ops.add and expr.red_op is ops.nullop:
        result = defaultdict(lambda: SEMIRING.zero)
        for term in expr.terms:
            for key, value in transpose(term).items():
                result[key] += value
        return result
    if expr.bin_op is ops.mul:
        assert len(expr.terms) == 2
        lhs, rhs = expr.terms
        lhs_adj = transpose(lhs)
        rhs_adj = transpose(rhs)
        result = defaultdict(lambda: SEMIRING.zero)
        for key, value in lhs_adj.items():
            result[key] += value * rhs
        for key, value in rhs_adj.items():
            result[key] += lhs * value
        return result
    raise NotImplementedError
