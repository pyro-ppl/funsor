from __future__ import absolute_import, division, print_function

import math
from collections import OrderedDict
from six.moves import reduce

import opt_einsum

import funsor.ops as ops
from funsor.cnf import Contraction
from funsor.delta import Delta, MultiDelta
from funsor.domains import find_domain
from funsor.gaussian import Gaussian, sym_inverse
from funsor.integrate import Integrate
from funsor.ops import AddOp, AssociativeOp, SubOp
from funsor.terms import Binary, Funsor, Number, Reduce, Unary, eager, moment_matching
from funsor.torch import Tensor


@eager.register(Binary, AddOp, Delta, (Number, Tensor, Gaussian))
def eager_add(op, delta, other):
    if delta.name in other.inputs:
        other = other(**{delta.name: delta.point})
        return op(delta, other)

    return None


@eager.register(Binary, SubOp, Delta, Gaussian)
def eager_sub(op, lhs, rhs):
    if lhs.name in rhs.inputs:
        rhs = rhs(**{lhs.name: lhs.point})
        return op(lhs, rhs)

    return None  # defer to default implementation


@eager.register(Binary, SubOp, MultiDelta, Gaussian)
def eager_add_delta_funsor(op, lhs, rhs):
    if lhs.fresh.intersection(rhs.inputs):
        rhs = rhs(**{name: point for name, point in lhs.terms if name in rhs.inputs})
        return op(lhs, rhs)

    return None  # defer to default implementation


#################################
# patterns for joint integration
#################################

@eager.register(Reduce, ops.AddOp, Unary, frozenset)
def eager_exp(op, arg, reduced_vars):
    if arg.op is ops.exp and isinstance(arg.arg, (Delta, MultiDelta)):
        return ops.exp(arg.arg.reduce(ops.logaddexp, reduced_vars))
    return None


@eager.register(Contraction, AssociativeOp, (ops.AddOp, AssociativeOp), frozenset, Tensor, Tensor)
def eager_contract(sum_op, prod_op, reduced_vars, lhs, rhs):
    if (sum_op, prod_op) == (ops.add, ops.mul):
        backend = "torch"
    elif (sum_op, prod_op) == (ops.logaddexp, ops.add):
        backend = "pyro.ops.einsum.torch_log"
    else:
        return prod_op(lhs, rhs).reduce(sum_op, reduced_vars)

    inputs = OrderedDict((k, d) for t in (lhs, rhs)
                         for k, d in t.inputs.items() if k not in reduced_vars)

    data = opt_einsum.contract(lhs.data, list(lhs.inputs),
                               rhs.data, list(rhs.inputs),
                               list(inputs), backend=backend)
    dtype = find_domain(prod_op, lhs.output, rhs.output).dtype
    return Tensor(data, inputs, dtype)


@moment_matching.register(Contraction, AssociativeOp, ops.AddOp, frozenset, (Number, Tensor), Gaussian)
def moment_matching_contract_joint(red_op, bin_op, reduced_vars, discrete, gaussian):

    if red_op is not ops.logaddexp:
        return None

    approx_vars = frozenset(k for k in reduced_vars if gaussian.inputs.get(k, 'real') != 'real')
    exact_vars = reduced_vars - approx_vars

    if exact_vars and approx_vars:
        return Contraction(red_op, bin_op, exact_vars, discrete, gaussian).reduce(red_op, approx_vars)

    if approx_vars and not exact_vars:
        new_discrete = discrete.reduce(ops.logaddexp, approx_vars.intersection(discrete.inputs))
        num_elements = reduce(ops.mul, [
            gaussian.inputs[k].num_elements for k in approx_vars.difference(discrete.inputs)], 1)
        if num_elements != 1:
            new_discrete -= math.log(num_elements)

        int_inputs = OrderedDict((k, d) for k, d in gaussian.inputs.items() if d.dtype != 'real')
        probs = (discrete - new_discrete).exp()
        old_loc = Tensor(gaussian.loc, int_inputs)
        new_loc = (probs * old_loc).reduce(ops.add, approx_vars)
        old_cov = Tensor(sym_inverse(gaussian.precision), int_inputs)
        diff = old_loc - new_loc
        outers = Tensor(diff.data.unsqueeze(-1) * diff.data.unsqueeze(-2), diff.inputs)
        new_cov = ((probs * old_cov).reduce(ops.add, approx_vars) +
                   (probs * outers).reduce(ops.add, approx_vars))
        new_precision = Tensor(sym_inverse(new_cov.data), new_cov.inputs)
        new_inputs = new_loc.inputs.copy()
        new_inputs.update((k, d) for k, d in gaussian.inputs.items() if d.dtype == 'real')
        new_gaussian = Gaussian(new_loc.data, new_precision.data, new_inputs)
        return new_discrete + new_gaussian

    return None


@eager.register(Contraction, ops.AddOp, ops.MulOp, frozenset, Unary, Funsor)
def eager_contraction_binary(red_op, bin_op, reduced_vars, lhs, rhs):
    if lhs.op is ops.exp and \
            isinstance(lhs.arg, (Delta, MultiDelta, Gaussian, Number, Tensor)) and \
            lhs.arg.fresh & reduced_vars:
        return Integrate(lhs.arg, rhs, reduced_vars)
    return None
