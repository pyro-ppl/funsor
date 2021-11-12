# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from collections.abc import Hashable

from funsor.cnf import Contraction
from funsor.interpretations import Interpretation, reflect
from funsor.interpreter import stack_reinterpret
from funsor.ops import AssociativeOp
from funsor.registry import KeyedRegistry
from funsor.terms import (
    Approximate,
    Binary,
    Cat,
    Funsor,
    Reduce,
    Scatter,
    Slice,
    Subs,
    substitute,
    to_funsor,
)

from . import instrument, interpreter, ops


def _alpha_unmangle(expr):
    alpha_subs = {
        name: name.split("__BOUND")[0] for name in expr.bound if "__BOUND" in name
    }
    if not alpha_subs:
        return tuple(expr._ast_values)

    return expr._alpha_convert(alpha_subs)


class AdjointTape(Interpretation):
    def __init__(self):
        super().__init__("adjoint")
        self.tape = []
        self._old_interpretation = None
        self._eager_to_lazy = {}

    def interpret(self, cls, *args):
        if cls in adjoint_ops:  # atomic op, don't trace internals
            with self._old_interpretation:
                result = cls(*args)
            self.tape.append((result, cls, args))
        else:
            result = self._old_interpretation.interpret(cls, *args)
        lazy_args = [
            self._eager_to_lazy.get(
                id(arg)
                if ops.is_numeric_array(arg) or not isinstance(arg, Hashable)
                else arg,
                arg,
            )
            for arg in args
        ]
        with self._old_interpretation:
            self._eager_to_lazy[result] = reflect.interpret(cls, *lazy_args)
        return result

    def __enter__(self):
        self.tape = []
        self._old_interpretation = interpreter.get_interpretation()
        return super().__enter__()

    def adjoint(self, sum_op, bin_op, root, targets=None, *, batch_vars=set()):
        zero = to_funsor(ops.UNITS[sum_op])
        one = to_funsor(ops.UNITS[bin_op])
        adjoint_values = defaultdict(lambda: zero)
        adjoint_values[root] = one

        reached_root = False
        while self.tape:
            output, fn, inputs = self.tape.pop()
            if not reached_root:
                if output is root:
                    reached_root = True
                else:
                    continue

            # reverse the effects of alpha-renaming
            with reflect:

                lazy_output = self._eager_to_lazy[output]
                lazy_fn = type(lazy_output)
                lazy_inputs = lazy_output._ast_values
                # TODO abstract this into a helper function
                # FIXME make lazy_output linear instead of quadratic in the size of the tape
                lazy_other_subs = tuple(
                    (name, to_funsor(name.split("__BOUND")[0], domain))
                    for name, domain in lazy_output.inputs.items()
                    if "__BOUND" in name
                )
                lazy_inputs = _alpha_unmangle(
                    substitute(lazy_fn(*lazy_inputs), lazy_other_subs)
                )
                lazy_output = type(lazy_output)(
                    *_alpha_unmangle(substitute(lazy_output, lazy_other_subs))
                )

                other_subs = tuple(
                    (name, to_funsor(name.split("__BOUND")[0], domain))
                    for name, domain in output.inputs.items()
                    if "__BOUND" in name
                )
                inputs = _alpha_unmangle(substitute(fn(*inputs), other_subs))
                output = type(output)(*_alpha_unmangle(substitute(output, other_subs)))

                self._eager_to_lazy[output] = lazy_output

            in_adjs = adjoint_ops(fn, sum_op, bin_op, adjoint_values[output], *inputs)
            for v, adjv in in_adjs:
                # Marginalize out message variables that don't appear in recipients.
                agg_vars = adjv.input_vars - v.input_vars - root.input_vars - batch_vars
                assert "particle" not in {var.name for var in agg_vars}  # DEBUG FIXME
                old_value = adjoint_values[v]
                adjoint_values[v] = sum_op(old_value, adjv.reduce(sum_op, agg_vars))

        result = defaultdict(lambda: zero)
        for key, value in adjoint_values.items():
            lazy_key = self._eager_to_lazy.get(key, key)
            result[lazy_key] = value

        if targets is None:
            return result
        return {target: result[target] for target in targets}


def forward_backward(sum_op, bin_op, expr, *, batch_vars=frozenset()):
    with AdjointTape() as tape:
        # TODO fix traversal order in AdjointTape instead of using stack_reinterpret
        forward = stack_reinterpret(expr)
    backward = tape.adjoint(sum_op, bin_op, forward, batch_vars=batch_vars)
    return forward, backward


def adjoint(sum_op, bin_op, expr):
    forward, backward = forward_backward(sum_op, bin_op, expr)
    return backward


# logaddexp/add
def _fail_default(*args):
    raise NotImplementedError("Should not be here! {}".format(args))


adjoint_ops = KeyedRegistry(default=_fail_default)
if instrument.DEBUG:
    adjoint_ops_register = adjoint_ops.register
    adjoint_ops.register = lambda *args: lambda fn: adjoint_ops_register(*args)(
        instrument.debug_logged(fn)
    )


@adjoint_ops.register(
    Binary, AssociativeOp, AssociativeOp, Funsor, AssociativeOp, Funsor, Funsor
)
def adjoint_binary(adj_sum_op, adj_prod_op, out_adj, op, lhs, rhs):
    if op is adj_prod_op:
        lhs_adj = adj_prod_op(out_adj, rhs)
        rhs_adj = adj_prod_op(out_adj, lhs)
        return ((lhs, lhs_adj), (rhs, rhs_adj))
    elif op is adj_sum_op:
        return ((lhs, out_adj), (rhs, out_adj))
    raise ValueError("should not be here!")


@adjoint_ops.register(
    Reduce, AssociativeOp, AssociativeOp, Funsor, AssociativeOp, Funsor, frozenset
)
def adjoint_reduce(adj_sum_op, adj_prod_op, out_adj, op, arg, reduced_vars):
    if op is adj_sum_op:
        out_adj = Approximate(
            adj_sum_op, out_adj, adj_prod_op(out_adj, arg), reduced_vars
        )
        return ((arg, out_adj),)
    elif op is adj_prod_op:  # plate!
        out = arg.reduce(adj_prod_op, reduced_vars)
        div_op = ops.SAFE_BINARY_INVERSES[adj_prod_op]
        return ((arg, div_op(adj_prod_op(out_adj, out), arg)),)
    raise ValueError("should not be here!")


@adjoint_ops.register(
    Contraction,
    AssociativeOp,
    AssociativeOp,
    Funsor,
    AssociativeOp,
    AssociativeOp,
    frozenset,
    Funsor,
)
def adjoint_contract_unary(
    adj_sum_op, adj_prod_op, out_adj, sum_op, prod_op, reduced_vars, arg
):
    return adjoint_reduce(adj_sum_op, adj_prod_op, out_adj, sum_op, arg, reduced_vars)


@adjoint_ops.register(
    Contraction,
    AssociativeOp,
    AssociativeOp,
    Funsor,
    AssociativeOp,
    AssociativeOp,
    frozenset,
    tuple,
)
def adjoint_contract_generic(
    adj_sum_op, adj_prod_op, out_adj, sum_op, prod_op, reduced_vars, terms
):
    assert len(terms) == 1 or len(terms) == 2
    return adjoint_ops(
        Contraction,
        adj_sum_op,
        adj_prod_op,
        out_adj,
        sum_op,
        prod_op,
        reduced_vars,
        *terms
    )


@adjoint_ops.register(
    Contraction,
    AssociativeOp,
    AssociativeOp,
    Funsor,
    AssociativeOp,
    AssociativeOp,
    frozenset,
    Funsor,
    Funsor,
)
def adjoint_contract(
    adj_sum_op, adj_prod_op, out_adj, sum_op, prod_op, reduced_vars, lhs, rhs
):
    if prod_op is adj_prod_op and sum_op in (ops.null, adj_sum_op):

        # the only change is here:
        out_adj = Approximate(
            adj_sum_op,
            out_adj,
            adj_prod_op(out_adj, adj_prod_op(lhs, rhs)),
            reduced_vars,
        )

        lhs_adj = adj_prod_op(out_adj, rhs)
        rhs_adj = adj_prod_op(lhs, out_adj)
        return ((lhs, lhs_adj), (rhs, rhs_adj))

    elif prod_op is adj_sum_op:
        if reduced_vars:
            raise NotImplementedError("TODO implement sum Contraction")
        return ((lhs, out_adj), (rhs, out_adj))

    raise ValueError("should not be here!")


@adjoint_ops.register(Cat, AssociativeOp, AssociativeOp, Funsor, str, tuple, str)
def adjoint_cat(adj_sum_op, adj_prod_op, out_adj, name, parts, part_name):
    if part_name not in out_adj.inputs:
        return tuple((part, out_adj) for part in parts)
    in_adjs = []
    start = 0
    size = sum(part.inputs[part_name].dtype for part in parts)
    for i, part in enumerate(parts):
        part_slice = Slice(name, start, start + part.inputs[part_name].dtype, 1, size)
        part_adj = out_adj(**{name: part_slice})
        in_adjs.append((part, part_adj))
        start += part.inputs[part_name].dtype
    return tuple(in_adjs)


@adjoint_ops.register(Subs, AssociativeOp, AssociativeOp, Funsor, Funsor, tuple)
def adjoint_subs(adj_sum_op, adj_prod_op, out_adj, arg, subs):

    # detect fresh variable collisions that should be relabeled and reduced
    relabel = {k: interpreter.gensym(k) for k, v in subs}
    relabeled_subs = tuple((relabel[k], v) for k, v in subs)
    relabeled_arg = arg(**relabel)

    reduced_vars = out_adj.input_vars - relabeled_arg.input_vars
    for k, v in subs:
        reduced_vars |= v.input_vars - relabeled_arg.input_vars

    relabeled_arg_adj = Scatter(adj_sum_op, relabeled_subs, out_adj, reduced_vars)
    arg_adj = relabeled_arg_adj(**{v: k for k, v in relabel.items()})
    return ((arg, arg_adj),)


@adjoint_ops.register(
    Scatter,
    AssociativeOp,
    AssociativeOp,
    Funsor,
    AssociativeOp,
    tuple,
    Funsor,
    frozenset,
)
def adjoint_scatter(adj_sum_op, adj_prod_op, out_adj, op, subs, source, reduced_vars):
    return ((source, out_adj(**dict(subs)).reduce(adj_sum_op, reduced_vars)),)
