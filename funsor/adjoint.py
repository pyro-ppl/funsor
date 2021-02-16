# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict, defaultdict

from funsor.cnf import Contraction, GaussianMixture, nullop
from funsor.domains import Bint
from funsor.gaussian import Gaussian, align_gaussian
from funsor.interpretations import Interpretation, reflect
from funsor.interpreter import stack_reinterpret
from funsor.ops import AssociativeOp
from funsor.registry import KeyedRegistry
from funsor.tensor import Tensor
from funsor.terms import (
    Binary,
    Cat,
    Funsor,
    Number,
    Reduce,
    Scatter,
    Slice,
    Subs,
    Variable,
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
        lazy_args = [self._eager_to_lazy.get(arg, arg) for arg in args]
        self._eager_to_lazy[result] = reflect.interpret(cls, *lazy_args)
        return result

    def __enter__(self):
        self.tape = []
        self._old_interpretation = interpreter.get_interpretation()
        return super().__enter__()

    def adjoint(self, sum_op, bin_op, root, targets=None):

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
            out_agg_vars = adjoint_values[output].input_vars - root.input_vars
            for v, adjv in in_adjs:
                agg_vars = out_agg_vars & (adjv.input_vars - v.input_vars)
                old_value = adjoint_values[v]
                adjoint_values[v] = sum_op(old_value, adjv.reduce(sum_op, agg_vars))

        result = defaultdict(lambda: zero)
        for key, value in adjoint_values.items():
            lazy_key = self._eager_to_lazy.get(key, key)
            result[lazy_key] = value

        if targets is None:
            return result
        return {target: result[target] for target in targets}


def adjoint(sum_op, bin_op, expr):
    with AdjointTape() as tape:
        # TODO fix traversal order in AdjointTape instead of using stack_reinterpret
        root = stack_reinterpret(expr)
    return tape.adjoint(sum_op, bin_op, root)


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
        return ((arg, out_adj),)
    elif op is adj_prod_op:  # plate!
        out = arg.reduce(adj_prod_op, reduced_vars)
        div_op = ops.SAFE_BINARY_INVERSES[adj_prod_op]
        return (
            (
                arg,
                div_op(adj_prod_op(out_adj, out), arg),
            ),
        )
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
    if prod_op is adj_prod_op and sum_op in (nullop, adj_sum_op):
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
    in_adjs = []
    start = 0
    size = sum(part.inputs[part_name].dtype for part in parts)
    for i, part in enumerate(parts):
        if part_name in out_adj.inputs:
            in_adjs.append(
                (
                    part,
                    out_adj(
                        **{
                            name: Slice(
                                name,
                                start,
                                start + part.inputs[part_name].dtype,
                                1,
                                size,
                            )
                        }
                    ),
                )
            )
            start += part.inputs[part_name].dtype
        else:
            in_adjs.append((part, out_adj))
    return tuple(in_adjs)


@adjoint_ops.register(Subs, AssociativeOp, AssociativeOp, Funsor, Funsor, tuple)
def adjoint_subs(adj_sum_op, adj_prod_op, out_adj, arg, subs):
    return ((arg, Scatter(adj_sum_op, subs, out_adj)),)


@adjoint_ops.register(
    Scatter,
    AssociativeOp,
    AssociativeOp,
    Funsor,
    AssociativeOp,
    tuple,
    Funsor,
)
def adjoint_scatter(adj_sum_op, adj_prod_op, out_adj, op, subs, source):
    return ((source, out_adj(**dict(subs))),)


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
            index.append(
                ops.new_arange(res.data, domain.dtype).reshape(
                    (-1,) + (1,) * offset_from_right
                )
            )

    # Construct a [:] slice for the output.
    for i, size in enumerate(res.output.shape):
        offset_from_right = len(res.output.shape) - i - 1
        index.append(
            ops.new_arange(res.data, size).reshape((-1,) + (1,) * offset_from_right)
        )

    # the only difference from Tensor.eager_subs is here:
    # instead of indexing the rhs (lhs = rhs[index]), we index the lhs (lhs[index] = rhs)

    # unsqueeze to make broadcasting work
    src_inputs, src_data = src.inputs.copy(), src.data
    for k, v in res.inputs.items():
        if k not in src.inputs and isinstance(subs[k], Number):
            src_inputs[k] = Bint[1]
            src_data = src_data.unsqueeze(-1 - len(src.output.shape))
    src = Tensor(src_data, src_inputs, src.output.dtype).align(tuple(res.inputs.keys()))

    # TODO refactor to scatter_add to support non-injective substitution
    # TODO add and test ops.scatter_add
    # data = ops.scatter_add(res.data, tuple(index), src.data)

    data = res.data
    data[tuple(index)] = src.data

    return Tensor(data, inputs, res.dtype)


@adjoint_ops.register(
    Subs, ops.LogaddexpOp, ops.AddOp, GaussianMixture, GaussianMixture, tuple
)
def adjoint_subs_gaussianmixture_gaussianmixture(
    adj_sum_op, adj_prod_op, out_adj, arg, subs
):

    if any(v.dtype == "real" and not isinstance(v, Variable) for k, v in subs):
        raise NotImplementedError(
            "TODO implement adjoint for substitution into Gaussian real variable"
        )

    # invert renaming
    renames = tuple((v.name, k) for k, v in subs if isinstance(v, Variable))
    out_adj = Subs(out_adj, renames)

    # inverting advanced indexing
    slices = tuple((k, v) for k, v in subs if not isinstance(v, Variable))

    assert len(slices + renames) == len(subs)

    in_adj_discrete = adjoint_ops(
        Subs, adj_sum_op, adj_prod_op, out_adj.terms[0], arg.terms[0], subs
    )[0][1]

    arg_int_inputs = OrderedDict(
        (k, v) for k, v in arg.inputs.items() if v.dtype != "real"
    )
    out_adj_int_inputs = OrderedDict(
        (k, v) for k, v in out_adj.inputs.items() if v.dtype != "real"
    )

    arg_real_inputs = OrderedDict(
        (k, v) for k, v in arg.inputs.items() if v.dtype == "real"
    )

    align_inputs = OrderedDict(
        (k, v) for k, v in out_adj.terms[1].inputs.items() if v.dtype != "real"
    )
    align_inputs.update(arg_real_inputs)
    out_adj_info_vec, out_adj_precision = align_gaussian(align_inputs, out_adj.terms[1])

    in_adj_info_vec = adjoint_ops(
        Subs,
        adj_sum_op,
        adj_prod_op,  # ops.add, ops.mul,
        Tensor(out_adj_info_vec, out_adj_int_inputs),
        Tensor(arg.terms[1].info_vec, arg_int_inputs),
        slices,
    )[0][1]

    in_adj_precision = adjoint_ops(
        Subs,
        adj_sum_op,
        adj_prod_op,  # ops.add, ops.mul,
        Tensor(out_adj_precision, out_adj_int_inputs),
        Tensor(arg.terms[1].precision, arg_int_inputs),
        slices,
    )[0][1]

    assert isinstance(in_adj_info_vec, Tensor)
    assert isinstance(in_adj_precision, Tensor)

    in_adj_gaussian = Gaussian(
        in_adj_info_vec.data, in_adj_precision.data, arg.inputs.copy()
    )

    in_adj = in_adj_gaussian + in_adj_discrete
    return ((arg, in_adj),)


@adjoint_ops.register(
    Subs, ops.LogaddexpOp, ops.AddOp, Gaussian, GaussianMixture, tuple
)
def adjoint_subs_gaussianmixture_discrete(adj_sum_op, adj_prod_op, out_adj, arg, subs):

    if any(v.dtype == "real" and not isinstance(v, Variable) for k, v in subs):
        raise NotImplementedError(
            "TODO implement adjoint for substitution into Gaussian real variable"
        )

    out_adj_int_inputs = OrderedDict(
        (k, v) for k, v in out_adj.inputs.items() if v.dtype != "real"
    )
    out_adj_ = out_adj + Tensor(
        out_adj.info_vec.new_zeros(out_adj.info_vec.shape[:-1]), out_adj_int_inputs
    )
    return adjoint_ops(Subs, adj_sum_op, adj_prod_op, out_adj_, arg, subs)


@adjoint_ops.register(
    Subs, ops.LogaddexpOp, ops.AddOp, (GaussianMixture, Gaussian), Gaussian, tuple
)
def adjoint_subs_gaussian_gaussian(adj_sum_op, adj_prod_op, out_adj, arg, subs):

    if any(v.dtype == "real" and not isinstance(v, Variable) for k, v in subs):
        raise NotImplementedError(
            "TODO implement adjoint for substitution into Gaussian real variable"
        )

    arg_int_inputs = OrderedDict(
        (k, v) for k, v in arg.inputs.items() if v.dtype != "real"
    )
    arg_ = arg + Tensor(arg.info_vec.new_zeros(arg.info_vec.shape[:-1]), arg_int_inputs)
    return (
        (arg, adjoint_ops(Subs, adj_sum_op, adj_prod_op, out_adj, arg_, subs)[0][1]),
    )


@adjoint_ops.register(
    Subs, ops.LogaddexpOp, ops.AddOp, (Number, Tensor), GaussianMixture, tuple
)
def adjoint_subs_gaussianmixture_discrete(adj_sum_op, adj_prod_op, out_adj, arg, subs):

    if any(v.dtype == "real" and not isinstance(v, Variable) for k, v in subs):
        raise NotImplementedError(
            "TODO implement adjoint for substitution into Gaussian real variable"
        )

    # invert renaming
    renames = tuple((v.name, k) for k, v in subs if isinstance(v, Variable))
    out_adj = Subs(out_adj, renames)

    # inverting advanced indexing
    slices = tuple((k, v) for k, v in subs if not isinstance(v, Variable))

    arg_int_inputs = OrderedDict(
        (k, v) for k, v in arg.inputs.items() if v.dtype != "real"
    )

    zeros_like_out = Subs(
        Tensor(
            arg.terms[1].info_vec.new_full(
                arg.terms[1].info_vec.shape[:-1], ops.UNITS[adj_prod_op]
            ),
            arg_int_inputs,
        ),
        slices,
    )
    out_adj = adj_prod_op(out_adj, zeros_like_out)

    in_adj_discrete = adjoint_ops(
        Subs, adj_sum_op, adj_prod_op, out_adj, arg.terms[0], subs
    )[0][1]

    # invert the slicing for the Gaussian term even though the message does not affect the values
    in_adj_info_vec = adjoint_ops(
        Subs,
        adj_sum_op,
        adj_prod_op,  # ops.add, ops.mul,
        zeros_like_out,
        Tensor(arg.terms[1].info_vec, arg_int_inputs),
        slices,
    )[0][1]

    in_adj_precision = adjoint_ops(
        Subs,
        adj_sum_op,
        adj_prod_op,  # ops.add, ops.mul,
        zeros_like_out,
        Tensor(arg.terms[1].precision, arg_int_inputs),
        slices,
    )[0][1]

    assert isinstance(in_adj_info_vec, Tensor)
    assert isinstance(in_adj_precision, Tensor)

    in_adj_gaussian = Gaussian(
        in_adj_info_vec.data, in_adj_precision.data, arg.inputs.copy()
    )

    in_adj = in_adj_gaussian + in_adj_discrete
    return ((arg, in_adj),)
