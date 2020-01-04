import math
from collections import OrderedDict
from functools import reduce
from typing import Tuple, Union

import torch
from multipledispatch import dispatch
from multipledispatch.variadic import Variadic

import funsor.ops as ops
from funsor.cnf import Contraction, GaussianMixture
from funsor.delta import Delta
from funsor.domains import bint
from funsor.gaussian import Gaussian, align_gaussian, cholesky_inverse
from funsor.ops import AssociativeOp
from funsor.terms import Funsor, Independent, Number, Reduce, Unary, eager, moment_matching, normalize
from funsor.torch import Tensor, align_tensor


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
        elif issubclass(type(part), GaussianMixture):  # TODO figure out why isinstance isn't working
            discrete, gaussian = part.terms[0], part.terms[1]
            discrete = align_tensor(int_inputs, discrete).expand(shape)
        else:
            raise NotImplementedError("TODO")
        discretes.append(discrete)
        info_vec, precision = align_gaussian(inputs, gaussian)
        info_vecs.append(info_vec.expand(shape + (-1,)))
        precisions.append(precision.expand(shape + (-1, -1)))
    if part_name != name:
        del inputs[part_name]
        del int_inputs[part_name]

    dim = 0
    info_vec = torch.cat(info_vecs, dim=dim)
    precision = torch.cat(precisions, dim=dim)
    inputs[name] = bint(info_vec.size(dim))
    int_inputs[name] = inputs[name]
    result = Gaussian(info_vec, precision, inputs)
    if any(d is not None for d in discretes):
        for i, d in enumerate(discretes):
            if d is None:
                discretes[i] = info_vecs[i].new_zeros(info_vecs[i].shape[:-1])
        discrete = torch.cat(discretes, dim=dim)
        result += Tensor(discrete, int_inputs)
    return result


#################################
# patterns for moment-matching
#################################

@moment_matching.register(Contraction, AssociativeOp, AssociativeOp, frozenset, Variadic[object])
def moment_matching_contract_default(*args):
    return None


@moment_matching.register(Contraction, ops.LogAddExpOp, ops.AddOp, frozenset, (Number, Tensor), Gaussian)
def moment_matching_contract_joint(red_op, bin_op, reduced_vars, discrete, gaussian):

    approx_vars = frozenset(k for k in reduced_vars if k in gaussian.inputs
                            and gaussian.inputs[k].dtype != 'real')
    exact_vars = reduced_vars - approx_vars

    if exact_vars and approx_vars:
        return Contraction(red_op, bin_op, exact_vars, discrete, gaussian).reduce(red_op, approx_vars)

    if approx_vars and not exact_vars:
        discrete += gaussian.log_normalizer
        new_discrete = discrete.reduce(ops.logaddexp, approx_vars.intersection(discrete.inputs))
        new_discrete = discrete.reduce(ops.logaddexp, approx_vars.intersection(discrete.inputs))
        num_elements = reduce(ops.mul, [
            gaussian.inputs[k].num_elements for k in approx_vars.difference(discrete.inputs)], 1)
        if num_elements != 1:
            new_discrete -= math.log(num_elements)

        int_inputs = OrderedDict((k, d) for k, d in gaussian.inputs.items() if d.dtype != 'real')
        probs = (discrete - new_discrete.clamp_finite()).exp()

        old_loc = Tensor(gaussian.info_vec.unsqueeze(-1).cholesky_solve(gaussian._precision_chol).squeeze(-1),
                         int_inputs)
        new_loc = (probs * old_loc).reduce(ops.add, approx_vars)
        old_cov = Tensor(cholesky_inverse(gaussian._precision_chol), int_inputs)
        diff = old_loc - new_loc
        outers = Tensor(diff.data.unsqueeze(-1) * diff.data.unsqueeze(-2), diff.inputs)
        new_cov = ((probs * old_cov).reduce(ops.add, approx_vars) +
                   (probs * outers).reduce(ops.add, approx_vars))

        # Numerically stabilize by adding bogus precision to empty components.
        total = probs.reduce(ops.add, approx_vars)
        mask = (total.data == 0).to(total.data.dtype).unsqueeze(-1).unsqueeze(-1)
        new_cov.data += mask * torch.eye(new_cov.data.size(-1))

        new_precision = Tensor(cholesky_inverse(ops.cholesky(new_cov.data)), new_cov.inputs)
        new_info_vec = new_precision.data.matmul(new_loc.data.unsqueeze(-1)).squeeze(-1)
        new_inputs = new_loc.inputs.copy()
        new_inputs.update((k, d) for k, d in gaussian.inputs.items() if d.dtype == 'real')
        new_gaussian = Gaussian(new_info_vec, new_precision.data, new_inputs)
        new_discrete -= new_gaussian.log_normalizer

        return new_discrete + new_gaussian

    return None


####################################################
# Patterns for normalizing
####################################################


@eager.register(Reduce, ops.AddOp, Unary[ops.ExpOp, Funsor], frozenset)
def eager_reduce_exp(op, arg, reduced_vars):
    # x.exp().reduce(ops.add) == x.reduce(ops.logaddexp).exp()
    log_result = arg.arg.reduce(ops.logaddexp, reduced_vars)
    if log_result is not normalize(Reduce, ops.logaddexp, arg.arg, reduced_vars):
        return log_result.exp()
    return None


@eager.register(Independent,
                (Contraction[ops.NullOp, ops.AddOp, frozenset, Tuple[Delta, Union[Number, Tensor], Gaussian]],
                 Contraction[ops.NullOp, ops.AddOp, frozenset, Tuple[Delta, Union[Number, Tensor, Gaussian]]]),
                str, str, str)
def eager_independent_joint(joint, reals_var, bint_var, diag_var):
    if diag_var not in joint.terms[0].fresh:
        return None

    delta = Independent(joint.terms[0], reals_var, bint_var, diag_var)
    new_terms = (delta,) + tuple(t.reduce(ops.add, bint_var) for t in joint.terms[1:])
    return reduce(joint.bin_op, new_terms)
