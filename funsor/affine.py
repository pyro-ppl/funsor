# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from functools import reduce, singledispatch

import opt_einsum

from funsor.interpreter import gensym
from funsor.tensor import Einsum, Tensor, get_default_prototype
from funsor.terms import Binary, Funsor, Lambda, Reduce, Unary, Variable, Bint

from . import ops


def is_affine(fn):
    """
    A sound but incomplete test to determine whether a funsor is affine with
    respect to all of its real inputs.

    :param Funsor fn: A funsor.
    :rtype: bool
    """
    return affine_inputs(fn) == _real_inputs(fn)


def _real_inputs(fn):
    return frozenset(k for k, d in fn.inputs.items() if d.dtype == "real")


def affine_inputs(fn):
    """
    Returns a [sound sub]set of real inputs of ``fn``
    wrt which ``fn`` is known to be affine.

    :param Funsor fn: A funsor.
    :return: A set of input names wrt which ``fn`` is affine.
    :rtype: frozenset
    """
    result = getattr(fn, '_affine_inputs', None)
    if result is None:
        result = fn._affine_inputs = _affine_inputs(fn)
    return result


@singledispatch
def _affine_inputs(fn):
    assert isinstance(fn, Funsor)
    return frozenset()


# Make registration public.
affine_inputs.register = _affine_inputs.register


@affine_inputs.register(Variable)
def _(fn):
    return _real_inputs(fn)


@affine_inputs.register(Unary)
def _(fn):
    if fn.op in (ops.neg, ops.add) or isinstance(fn.op, ops.ReshapeOp):
        return affine_inputs(fn.arg)
    return frozenset()


@affine_inputs.register(Binary)
def _(fn):
    if fn.op in (ops.add, ops.sub):
        return affine_inputs(fn.lhs) | affine_inputs(fn.rhs)
    if fn.op is ops.truediv:
        return affine_inputs(fn.lhs) - _real_inputs(fn.rhs)
    if isinstance(fn.op, ops.GetitemOp):
        return affine_inputs(fn.lhs)
    if fn.op in (ops.mul, ops.matmul):
        lhs_affine = affine_inputs(fn.lhs) - _real_inputs(fn.rhs)
        rhs_affine = affine_inputs(fn.rhs) - _real_inputs(fn.lhs)
        if not lhs_affine:
            return rhs_affine
        if not rhs_affine:
            return lhs_affine
        # This multilinear case introduces incompleteness, since some vars
        # could later be reduced, making remaining vars affine.
        return frozenset()
    return frozenset()


@affine_inputs.register(Reduce)
def _(fn):
    return affine_inputs(fn.arg) - fn.reduced_vars


@affine_inputs.register(Einsum)
def _(fn):
    # This is simply a multiary version of the above Binary(ops.mul, ...) case.
    results = []
    for i, x in enumerate(fn.operands):
        others = fn.operands[:i] + fn.operands[i+1:]
        other_inputs = reduce(ops.or_, map(_real_inputs, others), frozenset())
        results.append(affine_inputs(x) - other_inputs)
    # This multilinear case introduces incompleteness, since some vars
    # could later be reduced, making remaining vars affine.
    if sum(map(bool, results)) == 1:
        for result in results:
            if result:
                return result
    return frozenset()


def extract_affine(fn):
    """
    Extracts an affine representation of a funsor, satisfying::

        x = ...
        const, coeffs = extract_affine(x)
        y = sum(Einsum(eqn, (coeff, Variable(var, coeff.output)))
                for var, (coeff, eqn) in coeffs.items())
        assert_close(y, x)
        assert frozenset(coeffs) == affine_inputs(x)

    The ``coeffs`` will have one key per input wrt which ``fn`` is known to be
    affine (via :func:`affine_inputs` ), and ``const`` and ``coeffs.values``
    will all be constant wrt these inputs.

    The affine approximation is computed by ev evaluating ``fn`` at
    zero and each basis vector. To improve performance, users may want to run
    under the :func:`~funsor.memoize.memoize` interpretation.

    :param Funsor fn: A funsor that is affine wrt the (add,mul) semiring in
        some subset of its inputs.
    :return: A pair ``(const, coeffs)`` where const is a funsor with no real
        inputs and ``coeffs`` is an OrderedDict mapping input name to a
        ``(coefficient, eqn)`` pair in einsum form.
    :rtype: tuple
    """
    # NB: this depends on the global default backend.
    prototype = get_default_prototype()
    # Determine constant part by evaluating fn at zero.
    inputs = affine_inputs(fn)
    inputs = OrderedDict((k, v) for k, v in fn.inputs.items() if k in inputs)
    zeros = {k: Tensor(ops.new_zeros(prototype, v.shape)) for k, v in inputs.items()}
    const = fn(**zeros)

    # Determine linear coefficients by evaluating fn on basis vectors.
    name = gensym('probe')
    coeffs = OrderedDict()
    for k, v in inputs.items():
        dim = v.num_elements
        var = Variable(name, Bint[dim])
        subs = zeros.copy()
        subs[k] = Tensor(ops.new_eye(prototype, (dim,)).reshape((dim,) + v.shape))[var]
        coeff = Lambda(var, fn(**subs) - const).reshape(v.shape + const.shape)
        inputs1 = ''.join(map(opt_einsum.get_symbol, range(len(coeff.shape))))
        inputs2 = inputs1[:len(v.shape)]
        output = inputs1[len(v.shape):]
        eqn = '{},{}->{}'.format(inputs1, inputs2, output)
        coeffs[k] = coeff, eqn
    return const, coeffs


__all__ = [
    "affine_inputs",
    "extract_affine",
    "is_affine",
]
