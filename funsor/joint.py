# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
from collections import OrderedDict
from functools import reduce
from typing import Tuple, Union

from multipledispatch import dispatch

import funsor.ops as ops
from funsor.cnf import Contraction, GaussianMixture
from funsor.delta import Delta
from funsor.domains import Bint
from funsor.gaussian import Gaussian, align_gaussian
from funsor.interpretations import moment_matching, simplify
from funsor.tensor import Tensor, align_tensor
from funsor.terms import Funsor, Independent, Number, Reduce, Unary
from funsor.typing import Variadic


@dispatch(str, str, Variadic[(Gaussian, GaussianMixture)])
def eager_cat_homogeneous(name, part_name, *parts):
    assert parts
    output = parts[0].output
    inputs = OrderedDict([(part_name, None)])
    for part in parts:
        assert part.output == output
        assert part_name in part.inputs
        inputs.update(part.inputs)

    int_inputs = OrderedDict((k, v) for k, v in inputs.items() if v.dtype != "real")
    real_inputs = OrderedDict((k, v) for k, v in inputs.items() if v.dtype == "real")
    inputs = int_inputs.copy()
    inputs.update(real_inputs)
    discretes = []
    info_vecs = []
    precisions = []
    for part in parts:
        inputs[part_name] = part.inputs[part_name]
        int_inputs[part_name] = inputs[part_name]
        shape = tuple(d.size for d in int_inputs.values())
        if isinstance(part, Gaussian):
            discrete = None
            gaussian = part
        elif issubclass(
            type(part), GaussianMixture
        ):  # TODO figure out why isinstance isn't working
            discrete, gaussian = part.terms[0], part.terms[1]
            discrete = ops.expand(align_tensor(int_inputs, discrete), shape)
        else:
            raise NotImplementedError("TODO")
        discretes.append(discrete)
        info_vec, precision = align_gaussian(inputs, gaussian)
        info_vecs.append(ops.expand(info_vec, shape + (-1,)))
        precisions.append(ops.expand(precision, shape + (-1, -1)))
    if part_name != name:
        del inputs[part_name]
        del int_inputs[part_name]

    dim = 0
    info_vec = ops.cat(info_vecs, dim)
    precision = ops.cat(precisions, dim)
    inputs[name] = Bint[info_vec.shape[dim]]
    int_inputs[name] = inputs[name]
    result = Gaussian(info_vec, precision, inputs)
    if any(d is not None for d in discretes):
        for i, d in enumerate(discretes):
            if d is None:
                discretes[i] = ops.new_zeros(info_vecs[i], info_vecs[i].shape[:-1])
        discrete = ops.cat(discretes, dim)
        result = result + Tensor(discrete, int_inputs)
    return result


#################################
# patterns for moment-matching
#################################


@moment_matching.register(
    Contraction, ops.LogaddexpOp, ops.AddOp, frozenset, (Number, Tensor), Gaussian
)
def moment_matching_contract_joint(red_op, bin_op, reduced_vars, discrete, gaussian):
    approx_vars = frozenset(
        v for v in reduced_vars & gaussian.input_vars if v.dtype != "real"
    )
    if not approx_vars:
        return None

    exact_vars = reduced_vars - approx_vars
    if exact_vars:
        exact = Contraction(red_op, bin_op, exact_vars, discrete, gaussian)
        return exact.reduce(red_op, approx_vars)

    discrete += gaussian.log_normalizer
    new_discrete = discrete.reduce(ops.logaddexp, approx_vars & discrete.input_vars)
    num_elements = reduce(
        ops.mul, [v.output.num_elements for v in approx_vars - discrete.input_vars], 1
    )
    if num_elements != 1:
        new_discrete -= math.log(num_elements)

    int_inputs = OrderedDict(
        (k, d) for k, d in gaussian.inputs.items() if d.dtype != "real"
    )
    probs = (discrete - new_discrete.clamp_finite()).exp()

    old_loc = Tensor(
        ops.cholesky_solve(
            ops.unsqueeze(gaussian.info_vec, -1), gaussian._precision_chol
        ).squeeze(-1),
        int_inputs,
    )
    new_loc = (probs * old_loc).reduce(ops.add, approx_vars)
    old_cov = Tensor(ops.cholesky_inverse(gaussian._precision_chol), int_inputs)
    diff = old_loc - new_loc
    outers = Tensor(
        ops.unsqueeze(diff.data, -1) * ops.unsqueeze(diff.data, -2), diff.inputs
    )
    new_cov = (probs * old_cov).reduce(ops.add, approx_vars) + (probs * outers).reduce(
        ops.add, approx_vars
    )

    # Numerically stabilize by adding bogus precision to empty components.
    total = probs.reduce(ops.add, approx_vars)
    mask = ops.unsqueeze(ops.unsqueeze((total.data == 0), -1), -1)
    new_cov.data = new_cov.data + mask * ops.new_eye(
        new_cov.data, new_cov.data.shape[-1:]
    )

    new_precision = Tensor(
        ops.cholesky_inverse(ops.cholesky(new_cov.data)), new_cov.inputs
    )
    new_info_vec = (new_precision.data @ ops.unsqueeze(new_loc.data, -1)).squeeze(-1)
    new_inputs = new_loc.inputs.copy()
    new_inputs.update((k, d) for k, d in gaussian.inputs.items() if d.dtype == "real")
    new_gaussian = Gaussian(new_info_vec, new_precision.data, new_inputs)
    new_discrete -= new_gaussian.log_normalizer

    return new_discrete + new_gaussian


####################################################
# Patterns for normalizing
####################################################


@simplify.register(Reduce, ops.AddOp, Unary[ops.ExpOp, Funsor], frozenset)
def eager_reduce_exp(op, arg, reduced_vars):
    # x.exp().reduce(ops.add) == x.reduce(ops.logaddexp).exp()
    return arg.arg.reduce(ops.logaddexp, reduced_vars)


@simplify.register(
    Independent,
    (
        Contraction[
            ops.NullOp,
            ops.AddOp,
            frozenset,
            Tuple[Delta, Union[Number, Tensor], Gaussian],
        ],
        Contraction[
            ops.NullOp,
            ops.AddOp,
            frozenset,
            Tuple[Delta, Union[Number, Tensor, Gaussian]],
        ],
    ),
    str,
    str,
    str,
)
def eager_independent_joint(joint, reals_var, bint_var, diag_var):
    if diag_var not in joint.terms[0].fresh:
        return None

    delta = Independent(joint.terms[0], reals_var, bint_var, diag_var)
    new_terms = (delta,) + tuple(t.reduce(ops.add, bint_var) for t in joint.terms[1:])
    return reduce(joint.bin_op, new_terms)
