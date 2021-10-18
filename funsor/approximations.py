# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from functools import reduce, singledispatch

from funsor.cnf import Contraction, GaussianMixture
from funsor.delta import Delta
from funsor.gaussian import Gaussian, _compute_offsets
from funsor.instrument import debug_logged
from funsor.integrate import Integrate
from funsor.interpretations import DispatchedInterpretation
from funsor.tensor import Tensor, align_tensor
from funsor.terms import Approximate, Funsor
from funsor.typing import deep_isinstance

from . import ops

argmax_approximate = DispatchedInterpretation("argmax_approximate")
"""
Point-approximate at the argmax of the provided guide.
"""


@argmax_approximate.register(Approximate, ops.MaxOp, Funsor, Funsor, frozenset)
@argmax_approximate.register(Approximate, ops.LogaddexpOp, Funsor, Funsor, frozenset)
def argmax_approximate_logaddexp(op, model, guide, approx_vars):
    result = model
    argmax = compute_argmax(guide, approx_vars)
    for name, point in argmax.items():
        result += Delta(name, point)
    return result


mean_approximate = DispatchedInterpretation("mean_approximate")
"""
Point-approximate at the mean of the provided guide.
"""


@mean_approximate.register(Approximate, ops.LogaddexpOp, Funsor, Funsor, frozenset)
def mean_approximate_logaddexp(op, model, guide, approx_vars):
    results = [model]
    for var in approx_vars:
        var_mean = Integrate(guide, var, var)
        results.append(Delta(var.name, var_mean))
    return reduce(ops.add, results)


laplace_approximate = DispatchedInterpretation("laplace_approximate")
"""
Gaussian approximate using the value and Hessian of the model, evaluated at the
mode of the guide.
"""


@laplace_approximate.register(Approximate, ops.LogaddexpOp, Funsor, Funsor, frozenset)
def laplace_approximate_logaddexp(op, model, guide, approx_vars):
    point = compute_argmax(guide, approx_vars)
    hessian = compute_hessian(model, approx_vars)
    assert deep_isinstance(hessian, GaussianMixture)
    total = model(**point).reduce(ops.logaddexp, approx_vars)
    return total + hessian


################################################################################
# Computations.
# TODO Consider either making these Funsor methods or making .sample() and
# ._sample() singledispatch functions.


@singledispatch
def compute_hessian(model, approx_vars):
    # TODO adapt from moment_matching
    raise NotImplementedError


@singledispatch
def compute_argmax(model, approx_vars):
    """
    Computes argmax of a funsor.

    :param Funsor model: A function of the approximated vars.
    :param frozenset approx_vars: A frozenset of
        :class:`~funsor.terms.Variable` s to maximize.
    :returns: A dict mapping name (str) to point estimate (Funsor), for each
        variable name in ``approx_vars``.
    :rtype: str
    """
    if approx_vars.isdisjoint(model.input_vars):
        return {}  # nothing to do
    raise NotImplementedError


@compute_argmax.register(Tensor)
@debug_logged
def compute_argmax_tensor(model, approx_vars):
    approx_vars = approx_vars.intersection(model.input_vars)
    if not approx_vars:
        return {}

    # Partition inputs into batch_inputs + event_inputs.
    approx_names = frozenset(v.name for v in approx_vars)
    batch_inputs = OrderedDict()
    event_inputs = OrderedDict()
    for k, d in model.inputs.items():
        if k in approx_names:
            event_inputs[k] = d
        else:
            batch_inputs[k] = d
    inputs = batch_inputs.copy()
    inputs.update(event_inputs)
    data = align_tensor(inputs, model)

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
        result[name] = Tensor(mod_point % size, batch_inputs, size)
        mod_point = mod_point // size
    return result


@compute_argmax.register(Gaussian)
@debug_logged
def compute_argmax_gaussian(model, approx_vars):
    approx_vars = approx_vars.intersection(model.input_vars)
    if not approx_vars:
        return {}
    if any(v.dtype != "real" for v in approx_vars):
        raise ValueError(
            "Argmax of non-normalized Gaussian mixtures is intentionally "
            "not implemented. You probably want to normalize. To work around, "
            "add a zero Tensor/Array with given inputs."
        )

    # Partition inputs into int_inputs + real_inputs.
    int_inputs = OrderedDict()
    real_inputs = OrderedDict()
    for k, d in model.inputs.items():
        if d.dtype == "real":
            real_inputs[k] = d
        else:
            int_inputs[k] = d

    approx_names = frozenset(v.name for v in approx_vars)
    if approx_names == frozenset(real_inputs):
        mode = model._mean
        offsets, _ = _compute_offsets(real_inputs)
        result = {}
        for key, domain in real_inputs.items():
            data = mode[..., offsets[key] : offsets[key] + domain.num_elements]
            data = data.reshape(mode.shape[:-1] + domain.shape)
            result[key] = Tensor(data, int_inputs)
        return result

    raise NotImplementedError("TODO implement partial argmax of real variables")


@compute_argmax.register(GaussianMixture)
@debug_logged
def compute_argmax_gaussian_mixture(model, approx_vars):
    approx_vars = approx_vars.intersection(model.input_vars)
    if not approx_vars:
        return {}
    if any(model.reduced_vars):
        raise NotImplementedError
    discrete, gaussian = model.terms
    result = {}

    # Compute real argmax.
    real_vars = frozenset(v for v in gaussian.input_vars if v.dtype == "real")
    real_approx_vars = real_vars & approx_vars
    if real_approx_vars:
        result.update(compute_argmax(gaussian, real_approx_vars))

    # Compute int argmax.
    int_approx_vars = frozenset(
        v for v in model.input_vars & approx_vars if v.dtype != "real"
    )
    if int_approx_vars:
        discrete = discrete + gaussian.reduce(ops.logaddexp, real_vars)
        result.update(compute_argmax(discrete, int_approx_vars))

    return result


@compute_argmax.register(Contraction[ops.NullOp, ops.AddOp, frozenset, tuple])
def compute_argmax_contract(model, approx_vars):
    for t1 in model.terms:
        for t2 in model.terms:
            if t1 is t2:
                continue
            if not approx_vars.isdisjoint(t1.input_vars & t2.input_vars):
                raise ValueError("should never be here")

    result = {}
    for term in model.terms:
        result.update(compute_argmax(term, approx_vars & term.input_vars))
    return result


__all__ = [
    "argmax_approximate",
    "compute_argmax",
    "laplace_approximate",
    "mean_approximate",
]
