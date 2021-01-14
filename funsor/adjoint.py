# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict, defaultdict

import numpy as np

import funsor.interpreter as interpreter
import funsor.ops as ops
from funsor.cnf import Contraction, GaussianMixture, nullop
from funsor.domains import Bint
from funsor.gaussian import Gaussian, align_gaussian
from funsor.interpreter import interpretation
from funsor.ops import AssociativeOp
from funsor.registry import KeyedRegistry
from funsor.terms import Binary, Cat, Funsor, Number, Reduce, Slice, Subs, Variable, reflect, substitute, to_funsor
from funsor.tensor import Tensor


def _alpha_unmangle(expr):
    alpha_subs = {name: name.split("__BOUND")[0]
                  for name in expr.bound if "__BOUND" in name}
    if not alpha_subs:
        return tuple(expr._ast_values)

    return expr._alpha_convert(alpha_subs)


class AdjointTape(object):

    def __init__(self):
        self.tape = []
        self._old_interpretation = None

    def __call__(self, cls, *args):
        if cls in adjoint_ops:  # atomic op, don't trace internals
            with interpretation(self._old_interpretation):
                result = cls(*args)
            self.tape.append((result, cls, args))
        else:
            result = self._old_interpretation(cls, *args)
        return result

    def __enter__(self):
        self.tape = []
        self._old_interpretation = interpreter._INTERPRETATION
        interpreter.set_interpretation(self)
        return self

    def __exit__(self, *args):
        interpreter.set_interpretation(self._old_interpretation)
        self._old_interpretation = None

    def adjoint(self, red_op, bin_op, root, targets):

        bin_unit = to_funsor(ops.UNITS[bin_op])
        adjoint_values = defaultdict(lambda: bin_unit)

        reached_root = False
        while self.tape:
            output, fn, inputs = self.tape.pop()
            if not reached_root:
                if output is root:
                    reached_root = True
                else:
                    continue

            # reverse the effects of alpha-renaming
            with interpretation(reflect):
                other_subs = tuple((name, to_funsor(name.split("__BOUND")[0], domain))
                                   for name, domain in output.inputs.items() if "__BOUND" in name)
                inputs = _alpha_unmangle(substitute(fn(*inputs), other_subs))
                output = type(output)(*_alpha_unmangle(substitute(output, other_subs)))

            in_adjs = adjoint_ops(fn, red_op, bin_op, adjoint_values[output], *inputs)
            for v, adjv in in_adjs.items():
                adjoint_values[v] = bin_op(adjoint_values[v], adjv)

        target_adjs = {}
        for v in targets:
            target_adjs[v] = adjoint_values[v]
            if not isinstance(v, Variable):
                target_adjs[v] = bin_op(target_adjs[v], v)

        return target_adjs


# logaddexp/add
def _fail_default(*args):
    raise NotImplementedError("Should not be here! {}".format(args))


adjoint_ops = KeyedRegistry(default=_fail_default)
if interpreter._DEBUG:
    adjoint_ops_register = adjoint_ops.register
    adjoint_ops.register = lambda *args: lambda fn: adjoint_ops_register(*args)(interpreter.debug_logged(fn))


@adjoint_ops.register(Tensor, AssociativeOp, AssociativeOp, Funsor, (np.ndarray, np.generic), tuple, object)
def adjoint_tensor(adj_redop, adj_binop, out_adj, data, inputs, dtype):
    return {}


@adjoint_ops.register(Binary, AssociativeOp, AssociativeOp, Funsor, AssociativeOp, Funsor, Funsor)
def adjoint_binary(adj_redop, adj_binop, out_adj, op, lhs, rhs):
    assert (adj_redop, op) in ops.DISTRIBUTIVE_OPS

    lhs_adj = op(out_adj, rhs).reduce(adj_redop, rhs.input_vars - lhs.input_vars)
    rhs_adj = op(out_adj, lhs).reduce(adj_redop, lhs.input_vars - rhs.input_vars)

    return {lhs: lhs_adj, rhs: rhs_adj}


@adjoint_ops.register(Reduce, AssociativeOp, AssociativeOp, Funsor, AssociativeOp, Funsor, frozenset)
def adjoint_reduce(adj_redop, adj_binop, out_adj, op, arg, reduced_vars):
    assert adj_binop is op or (op, adj_binop) in ops.DISTRIBUTIVE_OPS

    if op is adj_redop:
        # XXX using a hack to simulate "expand"
        return {arg: adj_binop(out_adj, Binary(ops.PRODUCT_INVERSES[adj_binop], arg, arg))}
    elif op is adj_binop:  # plate!
        out = arg.reduce(op, reduced_vars)
        return {arg: adj_binop(out_adj, Binary(ops.PRODUCT_INVERSES[op], out, arg))}


@adjoint_ops.register(Contraction, AssociativeOp, AssociativeOp, Funsor,
                      AssociativeOp, AssociativeOp, frozenset, Funsor)
def adjoint_contract_unary(adj_redop, adj_binop, out_adj, sum_op, prod_op, reduced_vars, arg):
    return adjoint_reduce(adj_redop, adj_binop, out_adj, sum_op, arg, reduced_vars)


@adjoint_ops.register(Contraction, AssociativeOp, AssociativeOp, Funsor,
                      AssociativeOp, AssociativeOp, frozenset, tuple)
def adjoint_contract_generic(adj_redop, adj_binop, out_adj, sum_op, prod_op, reduced_vars, terms):
    assert len(terms) == 1 or len(terms) == 2
    return adjoint_ops(Contraction, adj_redop, adj_binop, out_adj, sum_op, prod_op, reduced_vars, *terms)


@adjoint_ops.register(Contraction, AssociativeOp, AssociativeOp, Funsor,
                      AssociativeOp, AssociativeOp, frozenset, Funsor, Funsor)
def adjoint_contract(adj_redop, adj_binop, out_adj, sum_op, prod_op, reduced_vars, lhs, rhs):
    assert sum_op is nullop or (sum_op, prod_op) in ops.DISTRIBUTIVE_OPS

    lhs_adj = Contraction(sum_op if sum_op is not nullop else adj_redop,
                          prod_op, rhs.input_vars - lhs.input_vars, out_adj, rhs)
    rhs_adj = Contraction(sum_op if sum_op is not nullop else adj_redop,
                          prod_op, lhs.input_vars - rhs.input_vars, out_adj, lhs)

    return {lhs: lhs_adj, rhs: rhs_adj}


@adjoint_ops.register(Cat, AssociativeOp, AssociativeOp, Funsor, str, tuple, str)
def adjoint_cat(adj_redop, adj_binop, out_adj, name, parts, part_name):
    in_adjs = {}
    start = 0
    size = sum(part.inputs[part_name].dtype for part in parts)
    for i, part in enumerate(parts):
        if part_name in out_adj.inputs:
            in_adjs[part] = out_adj(**{name: Slice(name, start, start + part.inputs[part_name].dtype, 1, size)})
            start += part.inputs[part_name].dtype
        else:
            in_adjs[part] = adj_binop(out_adj, Binary(ops.PRODUCT_INVERSES[adj_binop], part, part))
    return in_adjs


@adjoint_ops.register(Subs, AssociativeOp, AssociativeOp, (Number, Tensor), Tensor, tuple)
def adjoint_subs_tensor(adj_redop, adj_binop, out_adj, arg, subs):

    assert all(isinstance(v, Funsor) for k, v in subs)

    # invert renaming
    renames = tuple((v.name, k) for k, v in subs if isinstance(v, Variable))
    out_adj = Subs(out_adj, renames)

    # inverting advanced indexing
    slices = tuple((k, v) for k, v in subs if not isinstance(v, Variable))

    # TODO avoid reifying these zero/one tensors by using symbolic constants
    # ones for things that weren't sliced away
    ones_like_out = Subs(Tensor(ops.full_like(arg.data, ops.UNITS[adj_binop]),
                                arg.inputs.copy(), arg.output.dtype),
                         slices)
    arg_adj = adj_binop(out_adj, ones_like_out)

    # ones for things that were sliced away
    ones_like_arg = Tensor(ops.full_like(arg.data, ops.UNITS[adj_binop]),
                           arg.inputs.copy(), arg.output.dtype)
    arg_adj = _scatter(arg_adj, ones_like_arg, slices)

    return {arg: arg_adj}


def _scatter(src, res, subs):
    # inverse of advanced indexing
    # TODO check types of subs, in case some logic from eager_subs was accidentally left out?

    # use advanced indexing logic copied from Tensor.eager_subs:

    # materialize after checking for renaming case
    subs = OrderedDict((k, res.materialize(v)) for k, v in subs)

    # Compute result shapes.
    inputs = OrderedDict()
    for k, domain in res.inputs.items():
        inputs[k] = domain

    # Construct a dict with each input's positional dim,
    # counting from the right so as to support broadcasting.
    total_size = len(inputs) + len(res.output.shape)  # Assumes only scalar indices.
    new_dims = {}
    for k, domain in inputs.items():
        assert not domain.shape
        new_dims[k] = len(new_dims) - total_size

    # Use advanced indexing to construct a simultaneous substitution.
    index = []
    for k, domain in res.inputs.items():
        if k in subs:
            v = subs.get(k)
            if isinstance(v, Number):
                index.append(int(v.data))
            else:
                # Permute and expand v.data to end up at new_dims.
                assert isinstance(v, Tensor)
                v = v.align(tuple(k2 for k2 in inputs if k2 in v.inputs))
                assert isinstance(v, Tensor)
                v_shape = [1] * total_size
                for k2, size in zip(v.inputs, v.data.shape):
                    v_shape[new_dims[k2]] = size
                index.append(v.data.reshape(tuple(v_shape)))
        else:
            # Construct a [:] slice for this preserved input.
            offset_from_right = -1 - new_dims[k]
            index.append(ops.new_arange(res.data, domain.dtype).reshape(
                (-1,) + (1,) * offset_from_right))

    # Construct a [:] slice for the output.
    for i, size in enumerate(res.output.shape):
        offset_from_right = len(res.output.shape) - i - 1
        index.append(ops.new_arange(res.data, size).reshape(
            (-1,) + (1,) * offset_from_right))

    # the only difference from Tensor.eager_subs is here:
    # instead of indexing the rhs (lhs = rhs[index]), we index the lhs (lhs[index] = rhs)

    # unsqueeze to make broadcasting work
    src_inputs, src_data = src.inputs.copy(), src.data
    for k, v in res.inputs.items():
        if k not in src.inputs and isinstance(subs[k], Number):
            src_inputs[k] = Bint[1]
            src_data = src_data.unsqueeze(-1 - len(src.output.shape))
    src = Tensor(src_data, src_inputs, src.output.dtype).align(tuple(res.inputs.keys()))

    data = res.data
    data[tuple(index)] = src.data
    return Tensor(data, inputs, res.dtype)


@adjoint_ops.register(Subs, ops.LogaddexpOp, ops.AddOp, GaussianMixture, GaussianMixture, tuple)
def adjoint_subs_gaussianmixture_gaussianmixture(adj_redop, adj_binop, out_adj, arg, subs):

    if any(v.dtype == 'real' and not isinstance(v, Variable) for k, v in subs):
        raise NotImplementedError("TODO implement adjoint for substitution into Gaussian real variable")

    # invert renaming
    renames = tuple((v.name, k) for k, v in subs if isinstance(v, Variable))
    out_adj = Subs(out_adj, renames)

    # inverting advanced indexing
    slices = tuple((k, v) for k, v in subs if not isinstance(v, Variable))

    assert len(slices + renames) == len(subs)

    in_adj_discrete = adjoint_ops(Subs, adj_redop, adj_binop, out_adj.terms[0], arg.terms[0], subs)[arg.terms[0]]

    arg_int_inputs = OrderedDict((k, v) for k, v in arg.inputs.items() if v.dtype != 'real')
    out_adj_int_inputs = OrderedDict((k, v) for k, v in out_adj.inputs.items() if v.dtype != 'real')

    arg_real_inputs = OrderedDict((k, v) for k, v in arg.inputs.items() if v.dtype == 'real')

    align_inputs = OrderedDict((k, v) for k, v in out_adj.terms[1].inputs.items() if v.dtype != 'real')
    align_inputs.update(arg_real_inputs)
    out_adj_info_vec, out_adj_precision = align_gaussian(align_inputs, out_adj.terms[1])

    in_adj_info_vec = list(adjoint_ops(Subs, adj_redop, adj_binop,  # ops.add, ops.mul,
                                       Tensor(out_adj_info_vec, out_adj_int_inputs),
                                       Tensor(arg.terms[1].info_vec, arg_int_inputs),
                                       slices).values())[0]

    in_adj_precision = list(adjoint_ops(Subs, adj_redop, adj_binop,  # ops.add, ops.mul,
                                        Tensor(out_adj_precision, out_adj_int_inputs),
                                        Tensor(arg.terms[1].precision, arg_int_inputs),
                                        slices).values())[0]

    assert isinstance(in_adj_info_vec, Tensor)
    assert isinstance(in_adj_precision, Tensor)

    in_adj_gaussian = Gaussian(in_adj_info_vec.data, in_adj_precision.data, arg.inputs.copy())

    in_adj = in_adj_gaussian + in_adj_discrete
    return {arg: in_adj}


@adjoint_ops.register(Subs, ops.LogaddexpOp, ops.AddOp, Gaussian, GaussianMixture, tuple)
def adjoint_subs_gaussianmixture_discrete(adj_redop, adj_binop, out_adj, arg, subs):

    if any(v.dtype == 'real' and not isinstance(v, Variable) for k, v in subs):
        raise NotImplementedError("TODO implement adjoint for substitution into Gaussian real variable")

    out_adj_int_inputs = OrderedDict((k, v) for k, v in out_adj.inputs.items() if v.dtype != 'real')
    out_adj_ = out_adj + Tensor(out_adj.info_vec.new_zeros(out_adj.info_vec.shape[:-1]), out_adj_int_inputs)
    return {arg: adjoint_ops(Subs, adj_redop, adj_binop, out_adj_, arg, subs)[arg]}


@adjoint_ops.register(Subs, ops.LogaddexpOp, ops.AddOp, (GaussianMixture, Gaussian), Gaussian, tuple)
def adjoint_subs_gaussian_gaussian(adj_redop, adj_binop, out_adj, arg, subs):

    if any(v.dtype == 'real' and not isinstance(v, Variable) for k, v in subs):
        raise NotImplementedError("TODO implement adjoint for substitution into Gaussian real variable")

    arg_int_inputs = OrderedDict((k, v) for k, v in arg.inputs.items() if v.dtype != 'real')
    arg_ = arg + Tensor(arg.info_vec.new_zeros(arg.info_vec.shape[:-1]), arg_int_inputs)
    return {arg: adjoint_ops(Subs, adj_redop, adj_binop, out_adj, arg_, subs)[arg_]}


@adjoint_ops.register(Subs, ops.LogaddexpOp, ops.AddOp, (Number, Tensor), GaussianMixture, tuple)
def adjoint_subs_gaussianmixture_discrete(adj_redop, adj_binop, out_adj, arg, subs):

    if any(v.dtype == 'real' and not isinstance(v, Variable) for k, v in subs):
        raise NotImplementedError("TODO implement adjoint for substitution into Gaussian real variable")

    # invert renaming
    renames = tuple((v.name, k) for k, v in subs if isinstance(v, Variable))
    out_adj = Subs(out_adj, renames)

    # inverting advanced indexing
    slices = tuple((k, v) for k, v in subs if not isinstance(v, Variable))

    arg_int_inputs = OrderedDict((k, v) for k, v in arg.inputs.items() if v.dtype != 'real')

    zeros_like_out = Subs(Tensor(arg.terms[1].info_vec.new_full(arg.terms[1].info_vec.shape[:-1], ops.UNITS[adj_binop]),
                                 arg_int_inputs),
                          slices)
    out_adj = adj_binop(out_adj, zeros_like_out)

    in_adj_discrete = adjoint_ops(Subs, adj_redop, adj_binop, out_adj, arg.terms[0], subs)[arg.terms[0]]

    # invert the slicing for the Gaussian term even though the message does not affect the values
    in_adj_info_vec = list(adjoint_ops(Subs, adj_redop, adj_binop,  # ops.add, ops.mul,
                                       zeros_like_out,
                                       Tensor(arg.terms[1].info_vec, arg_int_inputs),
                                       slices).values())[0]

    in_adj_precision = list(adjoint_ops(Subs, adj_redop, adj_binop,  # ops.add, ops.mul,
                                        zeros_like_out,
                                        Tensor(arg.terms[1].precision, arg_int_inputs),
                                        slices).values())[0]

    assert isinstance(in_adj_info_vec, Tensor)
    assert isinstance(in_adj_precision, Tensor)

    in_adj_gaussian = Gaussian(in_adj_info_vec.data, in_adj_precision.data, arg.inputs.copy())

    in_adj = in_adj_gaussian + in_adj_discrete
    return {arg: in_adj}
