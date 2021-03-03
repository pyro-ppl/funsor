# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from functools import reduce, singledispatch

from funsor.cnf import GaussianMixture
from funsor.delta import Delta
from funsor.instrument import debug_logged
from funsor.integrate import Integrate
from funsor.interpretations import DispatchedInterpretation
from funsor.tensor import Tensor, align_tensor
from funsor.terms import Approximate, Funsor
from funsor.typing import deep_isinstance

from . import ops

"""
Point interpretation of :class:`~funsor.integrate.Integrate`
expressions. This falls back to the previous interpreter in other cases.
"""
argmax_approximate = DispatchedInterpretation("argmax_approximate")


@argmax_approximate.register(Approximate, ops.LogaddexpOp, Funsor, Funsor, frozenset)
def argmax_approximate_logaddexp(op, model, guide, approx_vars):
    result = model
    argmax = compute_argmax(guide, approx_vars)
    for name, point in argmax.items():
        result += Delta(name, point)
    return result


mean_approximate = DispatchedInterpretation("mean_approximate")


@mean_approximate.register(Approximate, ops.LogaddexpOp, Funsor, Funsor, frozenset)
def mean_approximate_logaddexp(op, model, guide, approx_vars):
    results = [model]
    for var in approx_vars:
        var_mean = Integrate(guide, var, var)
        results.append(Delta(var.name, var_mean))
    return reduce(ops.add, results)


laplace_approximate = DispatchedInterpretation("laplace_approximate")


@laplace_approximate.register(Approximate, ops.LogaddexpOp, Funsor, Funsor, frozenset)
def laplace_approximate_logaddexp(op, model, guide, approx_vars):
    point = compute_argmax(guide, approx_vars)
    hessian = compute_hessian(guide, approx_vars)
    assert deep_isinstance(hessian, GaussianMixture)
    total = (point + model).reduce(ops.LogaddexpOp, approx_vars)
    return total + (hessian - guide)


################################################################################
# Computations.


# TODO Consider either making this a Funsor method or making .sample() and
# .unscaled_sample() singledispatch functions.
@singledispatch
def compute_hessian(guide, approx_vars):
    raise NotImplementedError


# TODO Consider either making this a Funsor method or making .sample() and
# .unscaled_sample() singledispatch functions.
@singledispatch
def compute_argmax(guide, approx_vars):
    """
    :returns: A dict mapping
    """
    raise NotImplementedError


@compute_argmax.register(Tensor)
@debug_logged
def compute_argmax_tensor(guide, approx_vars):
    # Partition inputs into batch_inputs + event_inputs.
    approx_names = frozenset(v.name for v in approx_vars)
    batch_inputs = OrderedDict()
    event_inputs = OrderedDict()
    for k, d in guide.inputs.items():
        if k in approx_names:
            event_inputs[k] = d
        else:
            batch_inputs[k] = d
    inputs = batch_inputs.copy()
    inputs.update(event_inputs)
    data = align_tensor(inputs, guide)

    # Flatten and compute single argmax.
    batch_shape = data.shape[: len(batch_inputs)]
    flat_data = data.reshape(batch_shape + (-1,))
    flat_point = ops.argmax(flat_data, -1)
    assert flat_point.shape == batch_shape

    # Unflatten into deltas
    result = {}
    mod_point = flat_point
    for name, domain in reversed(list(event_inputs.items())):
        size = domain.dtype
        point = Tensor(mod_point % size, batch_inputs, size)
        mod_point = mod_point // size
        result[name] = point
    return result


__all__ = [
    "argmax_approximate",
    "compute_argmax",
    "laplace_approximate",
    "mean_approximate",
]
