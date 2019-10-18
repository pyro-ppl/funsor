from collections import OrderedDict
from functools import reduce

import opt_einsum
import torch
from multipledispatch import dispatch

from funsor.cnf import Contraction
from funsor.interpreter import gensym, interpretation
from funsor.terms import Binary, Funsor, Lambda, Reduce, Unary, Variable, bint, reflect
from funsor.torch import Einsum, Tensor

from . import ops


def is_affine(fn):
    """
    A sound but incomplete test to determine whether a funsor is affine with
    respect to all of its real inputs.

    :param Funsor fn: A funsor.
    :rtype: bool
    """
    assert isinstance(fn, Funsor)
    return _affine_inputs(fn) == _real_inputs(fn)


def _real_inputs(fn):
    return frozenset(k for k, d in fn.inputs.items() if d.dtype == "real")


@dispatch(Funsor)
def _affine_inputs(fn):
    """
    Returns a [sound sub]set of real inputs of ``fn``
    wrt which ``fn`` is known to be affine.
    """
    return frozenset()


@dispatch(Variable)
def _affine_inputs(fn):
    return _real_inputs(fn)


@dispatch(Unary)
def _affine_inputs(fn):
    if fn.op in (ops.neg, ops.add) or isinstance(fn.op, ops.ReshapeOp):
        return _affine_inputs(fn.arg)
    return frozenset()


@dispatch(Binary)
def _affine_inputs(fn):
    if fn.op in (ops.add, ops.sub):
        return _affine_inputs(fn.lhs) | _affine_inputs(fn.rhs)
    if fn.op is ops.truediv:
        return _affine_inputs(fn.lhs) - _real_inputs(fn.rhs)
    if isinstance(fn.op, ops.GetitemOp):
        return _affine_inputs(fn.lhs)
    if fn.op in (ops.mul, ops.matmul):
        lhs_affine = _affine_inputs(fn.lhs) - _real_inputs(fn.rhs)
        rhs_affine = _affine_inputs(fn.rhs) - _real_inputs(fn.lhs)
        if not lhs_affine:
            return rhs_affine
        if not rhs_affine:
            return lhs_affine
        # This multilinear case introduces incompleteness, since some vars
        # could later be reduced, making remaining vars affine.
        return frozenset()
    return frozenset()


@dispatch(Reduce)
def _affine_inputs(fn):
    return _affine_inputs(fn.arg) - fn.reduced_vars


@dispatch(Contraction)
def _affine_inputs(fn):
    with interpretation(reflect):
        flat = reduce(fn.bin_op, fn.terms).reduce(fn.red_op, fn.reduced_vars)
    return _affine_inputs(flat)


@dispatch(Einsum)
def _affine_inputs(fn):
    # This is simply a multiary version of the above Binary(ops.mul, ...) case.
    results = []
    for i, x in enumerate(fn.operands):
        others = fn.operands[:i] + fn.operands[i+1:]
        other_inputs = reduce(ops.or_, map(_real_inputs, others), frozenset())
        results.append(_affine_inputs(x) - other_inputs)
    # This multilinear case introduces incompleteness, since some vars
    # could later be reduced, making remaining vars affine.
    if sum(map(bool, results)) == 1:
        for result in results:
            if result:
                return result
    return frozenset()


def extract_affine(fn):
    """
    Extracts an affine representation of a funsor, which is exact for affine
    funsors and approximate otherwise. For affine funsors this satisfies::

        x = ...
        const, coeffs = extract_affine(x)
        y = sum(Einsum(eqn, (coeff, Variable(var, coeff.output)))
                for var, (coeff, eqn) in coeffs.items())
        assert_close(y, x)

    The affine approximation is computed by ev evaluating ``fn`` at
    zero and each basis vector. To improve performance, users may want to run
    under the :func:`~funsor.memoize.memoize` interpretation.

    :param Funsor fn: A funsor assumed to be affine wrt the (add,mul) semiring.
       The affine assumption is not checked.
    :return: A pair ``(const, coeffs)`` where const is a funsor with no real
        inputs and ``coeffs`` is an OrderedDict mapping input name to a
        ``(coefficient, eqn)`` pair in einsum form.
    :rtype: tuple
    """
    # Determine constant part by evaluating fn at zero.
    real_inputs = OrderedDict((k, v) for k, v in fn.inputs.items() if v.dtype == 'real')
    zeros = {k: Tensor(torch.zeros(v.shape)) for k, v in real_inputs.items()}
    const = fn(**zeros)

    # Determine linear coefficients by evaluating fn on basis vectors.
    name = gensym('probe')
    coeffs = OrderedDict()
    for k, v in real_inputs.items():
        dim = v.num_elements
        var = Variable(name, bint(dim))
        subs = zeros.copy()
        subs[k] = Tensor(torch.eye(dim).reshape((dim,) + v.shape))[var]
        coeff = Lambda(var, fn(**subs) - const).reshape(v.shape + const.shape)
        inputs1 = ''.join(map(opt_einsum.get_symbol, range(len(coeff.shape))))
        inputs2 = inputs1[:len(v.shape)]
        output = inputs1[len(v.shape):]
        eqn = f'{inputs1},{inputs2}->{output}'
        coeffs[k] = coeff, eqn
    return const, coeffs
