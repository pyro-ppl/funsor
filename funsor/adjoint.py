from collections import OrderedDict, defaultdict

import torch

import funsor.interpreter as interpreter
import funsor.ops as ops
from funsor.cnf import Contraction, nullop
from funsor.domains import bint
from funsor.interpreter import interpretation
from funsor.ops import AssociativeOp
from funsor.registry import KeyedRegistry
from funsor.terms import (
    Binary,
    Funsor,
    Number,
    Reduce,
    Subs,
    Variable,
    reflect,
    substitute,
    to_funsor,
)
from funsor.torch import Tensor, materialize


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
        with interpretation(self._old_interpretation):
            result = cls(*args)
        if issubclass(cls, (Reduce, Contraction, Binary, Tensor, Subs)):  # TODO make generic
            self.tape.append((result, cls, args))
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
        multiplicities = defaultdict(lambda: 0)

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
                multiplicities[v] += 1
                adjoint_values[v] = bin_op(adjoint_values[v], adjv)

        target_adjs = {}
        for v in targets:
            target_adjs[v] = adjoint_values[v]  # / multiplicities[v]  # TODO use correct op here with bin_op
            if not isinstance(v, Variable):
                target_adjs[v] = bin_op(target_adjs[v], v)

        return target_adjs


# logaddexp/add
def _fail_default(*args):
    raise NotImplementedError("Should not be here! {}".format(args))


adjoint_ops = KeyedRegistry(default=_fail_default)


@adjoint_ops.register(Tensor, AssociativeOp, AssociativeOp, Funsor, torch.Tensor, tuple, object)
def adjoint_tensor(adj_redop, adj_binop, out_adj, data, inputs, dtype):
    out = Tensor(data, inputs, dtype)
    all_vars = frozenset(k for (k, v) in inputs)
    in_adjs = {}
    for (k, v) in inputs:
        in_adj = adj_binop(out_adj, out).reduce(adj_redop, all_vars - {k})
        in_adjs[Variable(k, v)] = in_adj
    return in_adjs


@adjoint_ops.register(Binary, AssociativeOp, AssociativeOp, Funsor, AssociativeOp, Funsor, Funsor)
def adjoint_binary(adj_redop, adj_binop, out_adj, op, lhs, rhs):
    assert (adj_redop, op) in ops.DISTRIBUTIVE_OPS

    lhs_reduced_vars = frozenset(rhs.inputs) - frozenset(lhs.inputs)
    lhs_adj = op(out_adj, rhs).reduce(adj_redop, lhs_reduced_vars)

    rhs_reduced_vars = frozenset(lhs.inputs) - frozenset(rhs.inputs)
    rhs_adj = op(out_adj, lhs).reduce(adj_redop, rhs_reduced_vars)

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
def adjoint_contract_unary(adj_redop, adj_binop, out_adj, sum_op, prod_op, reduced_vars, terms):
    assert len(terms) == 1 or len(terms) == 2
    return adjoint_ops(Contraction, adj_redop, adj_binop, out_adj, sum_op, prod_op, reduced_vars, *terms)


@adjoint_ops.register(Contraction, AssociativeOp, AssociativeOp, Funsor,
                      AssociativeOp, AssociativeOp, frozenset, Funsor, Funsor)
def adjoint_contract(adj_redop, adj_binop, out_adj, sum_op, prod_op, reduced_vars, lhs, rhs):
    assert sum_op is nullop or (sum_op, prod_op) in ops.DISTRIBUTIVE_OPS

    lhs_reduced_vars = frozenset(rhs.inputs) - frozenset(lhs.inputs)
    lhs_adj = Contraction(sum_op if sum_op is not nullop else adj_redop, prod_op, lhs_reduced_vars, out_adj, rhs)

    rhs_reduced_vars = frozenset(lhs.inputs) - frozenset(rhs.inputs)
    rhs_adj = Contraction(sum_op if sum_op is not nullop else adj_redop, prod_op, rhs_reduced_vars, out_adj, lhs)

    return {lhs: lhs_adj, rhs: rhs_adj}


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
    ones_like_out = Subs(Tensor(torch.full_like(arg.data, ops.UNITS[adj_binop]),
                                arg.inputs.copy(), arg.output.dtype),
                         slices)
    arg_adj = adj_binop(out_adj, ones_like_out)

    # XXX ones? zeros? for things that were sliced away
    zeros_like_arg = Tensor(torch.full_like(arg.data, ops.UNITS[adj_binop]),
                            arg.inputs.copy(), arg.output.dtype)
    arg_adj = _scatter(arg_adj, zeros_like_arg, slices)

    return {arg: arg_adj}


def _scatter(src, res, subs):
    # inverse of advanced indexing
    assert isinstance(res, Tensor)
    assert isinstance(src, Tensor)
    # TODO check types of subs, in case some logic from eager_subs was accidentally left out?

    # use advanced indexing logic copied from Tensor.eager_subs:

    # materialize after checking for renaming case
    subs = OrderedDict((k, materialize(v)) for k, v in subs)

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
            index.append(torch.arange(domain.dtype).reshape(
                (-1,) + (1,) * offset_from_right))

    # Construct a [:] slice for the output.
    for i, size in enumerate(res.output.shape):
        offset_from_right = len(res.output.shape) - i - 1
        index.append(torch.arange(size).reshape(
            (-1,) + (1,) * offset_from_right))

    # the only difference from Tensor.eager_subs is here:
    # instead of indexing the rhs (lhs = rhs[index]), we index the lhs (lhs[index] = rhs)

    # unsqueeze to make broadcasting work
    src_inputs, src_data = src.inputs.copy(), src.data
    for k, v in res.inputs.items():
        if k not in src.inputs and isinstance(subs[k], Number):
            src_inputs[k] = bint(1)
            src_data = src_data.unsqueeze(-1)
    src = Tensor(src_data, src_inputs, src.output.dtype).align(tuple(res.inputs.keys()))

    data = res.data
    data[tuple(index)] = src.data
    return Tensor(data, inputs, res.dtype)
