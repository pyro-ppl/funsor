# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import funsor.ops as ops
from funsor.ops import AssociativeOp, LogOp
from funsor.terms import Binary, Reduce, Tuple, Unary, eager, lazy, Variable, Number, Lambda
from funsor.interpreter import interpretation
from funsor.domains import Bint, Real, Array, Reals
from collections import defaultdict
from functools import reduce, singledispatch
from funsor import Tensor


def to_var(x, name):
    var = Variable(name, Array["real", x.data.shape])[tuple(x.inputs)]
    return var


def to_arg(x):
    input_vars = tuple(Variable(key, value) for key, value in x.inputs.items())
    arg = reduce(lambda a, b: Lambda(b, a), reversed(input_vars), x)
    return arg


def fjit(cls, *args):
    new_args = []
    for field, arg in zip(cls._ast_fields, args):
        if isinstance(arg, (Number, Tensor)):
            arg = to_var(arg, field)
        new_args.append(arg)
    new_args = tuple(new_args)
    return cls(*new_args)


def linearize(cls, *args, log=True):
    jvp = logJVP if log else JVP
    new_args = []
    for arg_name, arg in zip(cls._ast_fields, args):
        if isinstance(arg, (Number, Tensor)):
            tangent_var = to_var(arg, "d" + arg_name)
            arg = jvp(arg, tangent_var)
        new_args.append(arg)
    new_args = tuple(new_args)
    return cls(*new_args)


def get_linear_terms(expr, linear_vars):
    breakpoint()
    pass


@singledispatch
def transpose(expr, linear_vars):
    get_linear_terms(expr, linear_vars)
    out_shape = tuple(value.size for key, value in expr.inputs.items() if key not in linear_vars)
    out_inputs = tuple(key for key in expr.inputs if key not in linear_vars)
    out_tangent = Variable("dout", Array["real", out_shape])[out_inputs]
    breakpoint()
    pass


class JVP(Tuple):
    """
    Tuple:(Primal, Tanget)
    Semiring: (Add, Mul)
    """
    sum_op = ops.add
    prod_op = ops.mul


class logJVP(Tuple):
    """
    Tuple: (LogPrimal, LogTanget)
    Semiring: (Logaddexp, Add)
    """
    sum_op = ops.logaddexp
    prod_op = ops.add


@eager.register(Binary, AssociativeOp, JVP, JVP)
@eager.register(Binary, AssociativeOp, logJVP, logJVP)
def jvp_binary(op, lhs, rhs):
    sum_op = lhs.sum_op
    prod_op = lhs.prod_op
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


@lazy.register(Reduce, AssociativeOp, JVP, frozenset)
@eager.register(Reduce, AssociativeOp, logJVP, frozenset)
def jvp_reduce(op, arg, reduced_vars):
    sum_op, prod_op = arg.sum_op, arg.prod_op
    arg_primal, arg_tangent = arg
    primal = Reduce(op, arg_primal, reduced_vars)
    if op is sum_op:
        tangent = Reduce(sum_op, arg_tangent, reduced_vars)
    elif op is prod_op:
        div_op = ops.SAFE_BINARY_INVERSES[prod_op]
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
