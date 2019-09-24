from collections import defaultdict

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
    Cat,
    Funsor,
    Reduce,
    Slice,
    Subs,
    Variable,
    lazy,
    to_funsor,
)
from funsor.torch import Tensor


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
            with interpretation(lazy):
                other_subs = {name: name.split("__BOUND")[0] for name in output.inputs if "__BOUND" in name}
                inputs = _alpha_unmangle(fn(*inputs)(**other_subs))
                output = type(output)(*_alpha_unmangle(output(**other_subs)))

            in_adjs = adjoint_ops(fn, red_op, bin_op, adjoint_values[output], *inputs)
            for v, adjv in in_adjs.items():
                multiplicities[v] += 1
                adjoint_values[v] = bin_op(adjoint_values[v], adjv)

        target_adjs = {}
        for v in targets:
            target_adjs[v] = adjoint_values[v] / multiplicities[v]  # TODO use correct op here with bin_op
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


@adjoint_ops.register(Subs, AssociativeOp, AssociativeOp, Funsor, Funsor, tuple)
def adjoint_subs_slice(adj_redop, adj_binop, out_adj, arg, subs):

    # XXX only handles slice and rename right now
    assert all(isinstance(v, (str, Variable, Slice)) for k, v in subs)

    # invert renaming
    renames = tuple((v.name, k) for k, v in subs if isinstance(v, Variable))
    arg_adj = Subs(out_adj, renames)

    # unslicing
    slices = tuple((k, v) for k, v in subs if isinstance(v, Slice))
    for k, s in slices:
        arg_adj = _unslice(arg_adj, k, s, arg.inputs[k].dtype)

    return {arg: arg_adj}


def _unslice(x, name, s, orig_size):
    assert isinstance(x, Funsor)  # XXX Tensor?
    assert isinstance(name, str)
    assert isinstance(s, Slice)

    def Zero(inputs):
        shape = sum([v.shape for v in inputs.values()], ())
        return Tensor(torch.zeros(shape), inputs)

    # crude version: repeat zeros like x
    # TODO do this in constant number of terms (one x and one zero)
    zeros_like_x = Zero(x.inputs.copy())
    center = Cat((x,) + (zeros_like_x,) * (s.step - 1), name)

    # TODO permute center correctly along input name

    remainder_start_inputs = x.inputs.copy()
    remainder_start_inputs[name] = bint(s.start)
    remainder_start = Zero(remainder_start_inputs)

    remainder_end_inputs = x.inputs.copy()
    remainder_end_inputs[name] = bint(orig_size - s.stop)
    remainder_end = Zero(remainder_end_inputs)

    terms = (remainder_start, center, remainder_end)
    xc = Cat(terms, name)

    return xc
