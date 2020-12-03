# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import re
from collections import OrderedDict, defaultdict
from functools import reduce
from math import gcd

import funsor.ops as ops
from funsor.cnf import Contraction
from funsor.domains import Bint
from funsor.ops import UNITS, AssociativeOp
from funsor.terms import Cat, Funsor, FunsorMeta, Number, Slice, Stack, Subs, Variable, eager, substitute, to_funsor
from funsor.util import quote


def _partition(terms, sum_vars):
    # Construct a bipartite graph between terms and the vars
    neighbors = OrderedDict([(t, []) for t in terms])
    for term in terms:
        for dim in term.inputs.keys():
            if dim in sum_vars:
                neighbors[term].append(dim)
                neighbors.setdefault(dim, []).append(term)

    # Partition the bipartite graph into connected components for contraction.
    components = []
    while neighbors:
        v, pending = neighbors.popitem()
        component = OrderedDict([(v, None)])  # used as an OrderedSet
        for v in pending:
            component[v] = None
        while pending:
            v = pending.pop()
            for v in neighbors.pop(v):
                if v not in component:
                    component[v] = None
                    pending.append(v)

        # Split this connected component into tensors and dims.
        component_terms = tuple(v for v in component if isinstance(v, Funsor))
        if component_terms:
            component_dims = frozenset(v for v in component if not isinstance(v, Funsor))
            components.append((component_terms, component_dims))
    return components


def _unroll_plate(factors, plate, size, var_to_ordinal):
    # replicated variables
    plate_vars = set()
    for var, ordinal in var_to_ordinal.items():
        if plate in ordinal:
            plate_vars.add(var)

    # unroll factors
    unrolled_factors = []
    for factor in factors:
        if plate in factor.inputs:
            unrolled_factors.extend([factor(
                    **{plate: i},
                    **{var: "{}_{}{}".format(var, plate, i) for var in plate_vars}
                ) for i in range(size)])
        else:
            unrolled_factors.append(factor)

    # unroll variables
    for var in plate_vars:
        ordinal = var_to_ordinal.pop(var)
        var_to_ordinal.update({"{}_{}{}".format(var, plate, i): ordinal.difference(plate)
                               for i in range(size)})

    return unrolled_factors, var_to_ordinal


def partial_unroll(factors, plates=frozenset()):
    """
    Performs partial unrolling of plated factor graphs to standard factor graphs.

    :return: a list of partially unrolled Funsors
        and a frozenset of partially unrolled variables.
    """
    assert isinstance(factors, (tuple, list))
    assert all(isinstance(f, Funsor) for f in factors)
    assert isinstance(plates, frozenset)

    var_to_ordinal = {}
    plate_to_size = {}
    for f in factors:
        ordinal = plates.intersection(f.inputs)
        for var in frozenset(f.inputs) - plates:
            var_to_ordinal[var] = var_to_ordinal.get(var, ordinal) & ordinal
        for plate in ordinal:
            plate_to_size[plate] = f.inputs[plate].size

    # unroll one plate at a time
    for plate in plates:
        size = plate_to_size[plate]
        factors, var_to_ordinal = _unroll_plate(factors, plate, size, var_to_ordinal)

    return factors, frozenset(var_to_ordinal.keys())


def partial_sum_product(sum_op, prod_op, factors, eliminate=frozenset(), plates=frozenset()):
    """
    Performs partial sum-product contraction of a collection of factors.

    :return: a list of partially contracted Funsors.
    :rtype: list
    """
    assert callable(sum_op)
    assert callable(prod_op)
    assert isinstance(factors, (tuple, list))
    assert all(isinstance(f, Funsor) for f in factors)
    assert isinstance(eliminate, frozenset)
    assert isinstance(plates, frozenset)
    sum_vars = eliminate - plates

    var_to_ordinal = {}
    ordinal_to_factors = defaultdict(list)
    for f in factors:
        ordinal = plates.intersection(f.inputs)
        ordinal_to_factors[ordinal].append(f)
        for var in sum_vars.intersection(f.inputs):
            var_to_ordinal[var] = var_to_ordinal.get(var, ordinal) & ordinal

    ordinal_to_vars = defaultdict(set)
    for var, ordinal in var_to_ordinal.items():
        ordinal_to_vars[ordinal].add(var)

    results = []
    while ordinal_to_factors:
        leaf = max(ordinal_to_factors, key=len)
        leaf_factors = ordinal_to_factors.pop(leaf)
        leaf_reduce_vars = ordinal_to_vars[leaf]
        for (group_factors, group_vars) in _partition(leaf_factors, leaf_reduce_vars):
            f = reduce(prod_op, group_factors).reduce(sum_op, group_vars)
            remaining_sum_vars = sum_vars.intersection(f.inputs)
            if not remaining_sum_vars:
                results.append(f.reduce(prod_op, leaf & eliminate))
            else:
                new_plates = frozenset().union(
                    *(var_to_ordinal[v] for v in remaining_sum_vars))
                if new_plates == leaf:
                    raise ValueError("intractable!")
                f = f.reduce(prod_op, leaf - new_plates)
                ordinal_to_factors[new_plates].append(f)

    return results


def modified_partial_sum_product(sum_op, prod_op, factors,
                                 eliminate=frozenset(), plate_to_step=dict()):
    """
    Generalization of the tensor variable elimination algorithm of
    :func:`funsor.sum_product.partial_sum_product` to handle markov dimensions
    in addition to plate dimensions. Markov dimensions are eliminated efficiently
    using the parallel-scan algorithm in :func:`funsor.sum_product.sequential_sum_product`.

    :param ~funsor.ops.AssociativeOp sum_op: A semiring sum operation.
    :param ~funsor.ops.AssociativeOp prod_op: A semiring product operation.
    :param factors: A collection of funsors.
    :type factors: tuple or list
    :param frozenset eliminate: A set of free variables to eliminate,
        including both sum variables and product variable.
    :param dict plate_to_step: A dict mapping markov dimensions to
        ``step`` dicts that map previous to current variable name.
        Plates are passed with an empty ``step``.
    :return: a list of partially contracted Funsors.
    :rtype: list
    """
    assert callable(sum_op)
    assert callable(prod_op)
    assert isinstance(factors, (tuple, list))
    assert all(isinstance(f, Funsor) for f in factors)
    assert isinstance(eliminate, frozenset)
    assert isinstance(plate_to_step, dict)
    plates = frozenset(plate_to_step.keys())
    sum_vars = eliminate - plates
    prod_vars = eliminate.intersection(plates)
    markov_sum_vars = frozenset()
    for step in plate_to_step.values():
        markov_sum_vars |= frozenset(step.keys()) | frozenset(step.values())
    markov_sum_vars &= sum_vars
    markov_prod_vars = frozenset(k for k, v in plate_to_step.items() if v and k in eliminate)
    markov_sum_to_prod = defaultdict(set)
    for markov_prod in markov_prod_vars:
        for k, v in plate_to_step[markov_prod].items():
            markov_sum_to_prod[k].add(markov_prod)
            markov_sum_to_prod[v].add(markov_prod)

    var_to_ordinal = {}
    ordinal_to_factors = defaultdict(list)
    for f in factors:
        ordinal = plates.intersection(f.inputs)
        ordinal_to_factors[ordinal].append(f)
        for var in sum_vars.intersection(f.inputs):
            var_to_ordinal[var] = var_to_ordinal.get(var, ordinal) & ordinal

    ordinal_to_vars = defaultdict(set)
    for var, ordinal in var_to_ordinal.items():
        ordinal_to_vars[ordinal].add(var)

    results = []
    while ordinal_to_factors:
        leaf = max(ordinal_to_factors, key=len)
        leaf_factors = ordinal_to_factors.pop(leaf)
        leaf_reduce_vars = ordinal_to_vars[leaf]
        for (group_factors, group_vars) in _partition(leaf_factors, leaf_reduce_vars | markov_prod_vars):
            # eliminate non markov vars
            nonmarkov_vars = group_vars - markov_sum_vars - markov_prod_vars
            f = reduce(prod_op, group_factors).reduce(sum_op, nonmarkov_vars)
            # eliminate markov vars
            markov_vars = group_vars.intersection(markov_sum_vars)
            if markov_vars:
                markov_prod_var = [markov_sum_to_prod[var] for var in markov_vars]
                assert all(p == markov_prod_var[0] for p in markov_prod_var)
                if len(markov_prod_var[0]) != 1:
                    raise ValueError("intractable!")
                time = next(iter(markov_prod_var[0]))
                for v in sum_vars.intersection(f.inputs):
                    if time in var_to_ordinal[v] and var_to_ordinal[v] < leaf:
                        raise ValueError("intractable!")
                time_var = Variable(time, f.inputs[time])
                group_step = {k: v for (k, v) in plate_to_step[time].items() if v in markov_vars}
                f = MarkovProduct(sum_op, prod_op, f, time_var, group_step)
                f = f.reduce(sum_op, frozenset(group_step.keys()) | frozenset(group_step.values()))

            remaining_sum_vars = sum_vars.intersection(f.inputs)

            if not remaining_sum_vars:
                results.append(f.reduce(prod_op, leaf & prod_vars - markov_prod_vars))
            else:
                new_plates = frozenset().union(
                    *(var_to_ordinal[v] for v in remaining_sum_vars))
                if new_plates == leaf:
                    raise ValueError("intractable!")
                f = f.reduce(prod_op, leaf - new_plates - markov_prod_vars)
                ordinal_to_factors[new_plates].append(f)

    return results


def sum_product(sum_op, prod_op, factors, eliminate=frozenset(), plates=frozenset()):
    """
    Performs sum-product contraction of a collection of factors.

    :return: a single contracted Funsor.
    :rtype: :class:`~funsor.terms.Funsor`
    """
    factors = partial_sum_product(sum_op, prod_op, factors, eliminate, plates)
    return reduce(prod_op, factors, Number(UNITS[prod_op]))


def naive_sequential_sum_product(sum_op, prod_op, trans, time, step):
    assert isinstance(sum_op, AssociativeOp)
    assert isinstance(prod_op, AssociativeOp)
    assert isinstance(trans, Funsor)
    assert isinstance(time, Variable)
    assert isinstance(step, dict)
    assert all(isinstance(k, str) for k in step.keys())
    assert all(isinstance(v, str) for v in step.values())
    if time.name in trans.inputs:
        assert time.output == trans.inputs[time.name]

    step = OrderedDict(sorted(step.items()))
    drop = tuple("_drop_{}".format(i) for i in range(len(step)))
    prev_to_drop = dict(zip(step.keys(), drop))
    curr_to_drop = dict(zip(step.values(), drop))
    drop = frozenset(drop)

    time, duration = time.name, time.output.size
    factors = [trans(**{time: t}) for t in range(duration)]
    while len(factors) > 1:
        y = factors.pop()(**prev_to_drop)
        x = factors.pop()(**curr_to_drop)
        xy = prod_op(x, y).reduce(sum_op, drop)
        factors.append(xy)
    return factors[0]


def sequential_sum_product(sum_op, prod_op, trans, time, step):
    """
    For a funsor ``trans`` with dimensions ``time``, ``prev`` and ``curr``,
    computes a recursion equivalent to::

        tail_time = 1 + arange("time", trans.inputs["time"].size - 1)
        tail = sequential_sum_product(sum_op, prod_op,
                                      trans(time=tail_time),
                                      time, {"prev": "curr"})
        return prod_op(trans(time=0)(curr="drop"), tail(prev="drop")) \
           .reduce(sum_op, "drop")

    but does so efficiently in parallel in O(log(time)).

    :param ~funsor.ops.AssociativeOp sum_op: A semiring sum operation.
    :param ~funsor.ops.AssociativeOp prod_op: A semiring product operation.
    :param ~funsor.terms.Funsor trans: A transition funsor.
    :param Variable time: The time input dimension.
    :param dict step: A dict mapping previous variables to current variables.
        This can contain multiple pairs of prev->curr variable names.
    """
    assert isinstance(sum_op, AssociativeOp)
    assert isinstance(prod_op, AssociativeOp)
    assert isinstance(trans, Funsor)
    assert isinstance(time, Variable)
    assert isinstance(step, dict)
    assert all(isinstance(k, str) for k in step.keys())
    assert all(isinstance(v, str) for v in step.values())
    if time.name in trans.inputs:
        assert time.output == trans.inputs[time.name]

    step = OrderedDict(sorted(step.items()))
    drop = tuple("_drop_{}".format(i) for i in range(len(step)))
    prev_to_drop = dict(zip(step.keys(), drop))
    curr_to_drop = dict(zip(step.values(), drop))
    drop = frozenset(drop)

    time, duration = time.name, time.output.size
    while duration > 1:
        even_duration = duration // 2 * 2
        x = trans(**{time: Slice(time, 0, even_duration, 2, duration)}, **curr_to_drop)
        y = trans(**{time: Slice(time, 1, even_duration, 2, duration)}, **prev_to_drop)
        contracted = Contraction(sum_op, prod_op, drop, x, y)

        if duration > even_duration:
            extra = trans(**{time: Slice(time, duration - 1, duration)})
            contracted = Cat(time, (contracted, extra))
        trans = contracted
        duration = (duration + 1) // 2
    return trans(**{time: 0})


def mixed_sequential_sum_product(sum_op, prod_op, trans, time, step, num_segments=None):
    """
    For a funsor ``trans`` with dimensions ``time``, ``prev`` and ``curr``,
    computes a recursion equivalent to::

        tail_time = 1 + arange("time", trans.inputs["time"].size - 1)
        tail = sequential_sum_product(sum_op, prod_op,
                                      trans(time=tail_time),
                                      time, {"prev": "curr"})
        return prod_op(trans(time=0)(curr="drop"), tail(prev="drop")) \
           .reduce(sum_op, "drop")

    by mixing parallel and serial scan algorithms over ``num_segments`` segments.

    :param ~funsor.ops.AssociativeOp sum_op: A semiring sum operation.
    :param ~funsor.ops.AssociativeOp prod_op: A semiring product operation.
    :param ~funsor.terms.Funsor trans: A transition funsor.
    :param Variable time: The time input dimension.
    :param dict step: A dict mapping previous variables to current variables.
        This can contain multiple pairs of prev->curr variable names.
    :param int num_segments: number of segments for the first stage
    """
    time_var, time, duration = time, time.name, time.output.size
    num_segments = duration if num_segments is None else num_segments
    assert num_segments > 0 and duration > 0

    # handle unevenly sized segments by chopping off the final segment and calling mixed_sequential_sum_product again
    if duration % num_segments and duration - duration % num_segments > 0:
        remainder = trans(**{time: Slice(time, duration - duration % num_segments, duration, 1, duration)})
        initial = trans(**{time: Slice(time, 0, duration - duration % num_segments, 1, duration)})
        initial_eliminated = mixed_sequential_sum_product(
            sum_op, prod_op, initial, Variable(time, Bint[duration - duration % num_segments]), step,
            num_segments=num_segments)
        final = Cat(time, (Stack(time, (initial_eliminated,)), remainder))
        final_eliminated = naive_sequential_sum_product(
            sum_op, prod_op, final, Variable(time, Bint[1 + duration % num_segments]), step)
        return final_eliminated

    # handle degenerate cases that reduce to a single stage
    if num_segments == 1:
        return naive_sequential_sum_product(sum_op, prod_op, trans, time_var, step)
    if num_segments >= duration:
        return sequential_sum_product(sum_op, prod_op, trans, time_var, step)

    # break trans into num_segments segments of equal length
    segment_length = duration // num_segments
    segments = [trans(**{time: Slice(time, i * segment_length, (i + 1) * segment_length, 1, duration)})
                for i in range(num_segments)]

    first_stage_result = naive_sequential_sum_product(
        sum_op, prod_op, Stack(time + "__SEGMENTED", tuple(segments)),
        Variable(time, Bint[segment_length]), step)

    second_stage_result = sequential_sum_product(
        sum_op, prod_op, first_stage_result,
        Variable(time + "__SEGMENTED", Bint[num_segments]), step)

    return second_stage_result


def naive_sarkka_bilmes_product(sum_op, prod_op, trans, time_var, global_vars=frozenset()):

    assert isinstance(global_vars, frozenset)

    time = time_var.name

    def get_shift(name):
        return len(re.search("^P*", name).group(0))

    def shift_name(name, t):
        return t * "P" + name

    def shift_funsor(f, t):
        if t == 0:
            return f
        return f(**{name: shift_name(name, t) for name in f.inputs
                    if name != time and name not in global_vars})

    lags = {get_shift(name) for name in trans.inputs if name != time}
    lags.discard(0)
    if not lags:
        return naive_sequential_sum_product(sum_op, prod_op, trans, time_var, {})

    period = int(reduce(lambda a, b: a * b // gcd(a, b), list(lags)))

    duration = trans.inputs[time].size
    if duration % period:
        raise NotImplementedError("TODO handle partial windows")

    result = trans(**{time: duration - 1})
    original_names = frozenset(name for name in trans.inputs
                               if name != time and name not in global_vars
                               and not name.startswith("P"))
    for t in range(trans.inputs[time].size - 2, -1, -1):
        result = prod_op(shift_funsor(trans(**{time: t}), duration - t - 1), result)
        sum_vars = frozenset(shift_name(name, duration - t - 1) for name in original_names)
        result = result.reduce(sum_op, sum_vars)

    result = result(**{name: name.replace("P" * duration, "P") for name in result.inputs})
    return result


def sarkka_bilmes_product(sum_op, prod_op, trans, time_var, global_vars=frozenset(), num_periods=1):

    assert isinstance(global_vars, frozenset)

    time = time_var.name

    def get_shift(name):
        return len(re.search("^P*", name).group(0))

    def shift_name(name, t):
        return t * "P" + name

    def shift_funsor(f, t):
        if t == 0:
            return f
        return f(**{name: shift_name(name, t) for name in f.inputs
                    if name != time and name not in global_vars})

    lags = {get_shift(name) for name in trans.inputs if name != time}
    lags.discard(0)
    if not lags:
        return sequential_sum_product(sum_op, prod_op, trans, time_var, {})

    period = int(reduce(lambda a, b: a * b // gcd(a, b), list(lags)))
    original_names = frozenset(name for name in trans.inputs
                               if name != time and name not in global_vars
                               and not name.startswith("P"))
    renamed_factors = []
    duration = trans.inputs[time].size
    if duration % period:
        raise NotImplementedError("TODO handle partial windows")

    for t in range(period):
        slice_t = Slice(time, t, duration - period + t + 1, period, duration)
        factor = shift_funsor(trans, period - t - 1)
        factor = factor(**{time: slice_t})
        renamed_factors.append(factor)

    block_trans = reduce(prod_op, renamed_factors)
    block_step = {shift_name(name, period): name for name in block_trans.inputs
                  if name != time and name not in global_vars and get_shift(name) < period}
    block_time_var = Variable(time_var.name, Bint[duration // period])
    final_chunk = mixed_sequential_sum_product(
        sum_op, prod_op, block_trans, block_time_var, block_step,
        num_segments=max(1, duration // (period * num_periods)))
    final_sum_vars = frozenset(
        shift_name(name, t) for name in original_names for t in range(1, period))
    result = final_chunk.reduce(sum_op, final_sum_vars)
    result = result(**{name: name.replace("P" * period, "P") for name in result.inputs})
    return result


class MarkovProductMeta(FunsorMeta):
    """
    Wrapper to convert ``step`` to a tuple and fill in default ``step_names``.
    """
    def __call__(cls, sum_op, prod_op, trans, time, step, step_names=None):
        if isinstance(time, str):
            assert time in trans.inputs, "please pass Variable(time, ...)"
            time = Variable(time, trans.inputs[time])
        if isinstance(step, dict):
            step = frozenset(step.items())
        if step_names is None:
            step_names = frozenset((k, k) for pair in step for k in pair)
        if isinstance(step_names, dict):
            step_names = frozenset(step_names.items())
        return super().__call__(sum_op, prod_op, trans, time, step, step_names)


class MarkovProduct(Funsor, metaclass=MarkovProductMeta):
    """
    Lazy representation of :func:`sequential_sum_product` .

    :param AssociativeOp sum_op: A marginalization op.
    :param AssociativeOp prod_op: A Bayesian fusion op.
    :param Funsor trans: A sequence of transition factors,
        usually varying along the ``time`` input.
    :param time: A time dimension.
    :type time: str or Variable
    :param dict step: A str-to-str mapping of "previous" inputs of ``trans``
        to "current" inputs of ``trans``.
    :param dict step_names: Optional, for internal use by alpha conversion.
    """
    def __init__(self, sum_op, prod_op, trans, time, step, step_names):
        assert isinstance(sum_op, AssociativeOp)
        assert isinstance(prod_op, AssociativeOp)
        assert isinstance(trans, Funsor)
        assert isinstance(time, Variable)
        assert isinstance(step, frozenset)
        assert isinstance(step_names, frozenset)
        step = dict(step)
        step_names = dict(step_names)
        assert all(isinstance(k, str) for k in step_names.keys())
        assert all(isinstance(v, str) for v in step_names.values())
        assert set(step_names) == set(step).union(step.values())
        inputs = OrderedDict((step_names.get(k, k), v)
                             for k, v in trans.inputs.items()
                             if k != time.name)
        output = trans.output
        fresh = frozenset(step_names.values())
        bound = frozenset(step_names.keys()) | {time.name}
        super().__init__(inputs, output, fresh, bound)
        self.sum_op = sum_op
        self.prod_op = prod_op
        self.trans = trans
        self.time = time
        self.step = step
        self.step_names = step_names

    def _alpha_convert(self, alpha_subs):
        assert self.bound.issuperset(alpha_subs)
        time = Variable(alpha_subs.get(self.time.name, self.time.name),
                        self.time.output)
        step = frozenset((alpha_subs.get(k, k), alpha_subs.get(v, v))
                         for k, v in self.step.items())
        step_names = frozenset((alpha_subs.get(k, k), v)
                               for k, v in self.step_names.items())
        alpha_subs = {k: to_funsor(v, self.trans.inputs[k])
                      for k, v in alpha_subs.items()
                      if k in self.trans.inputs}
        trans = substitute(self.trans, alpha_subs)
        return self.sum_op, self.prod_op, trans, time, step, step_names

    def eager_subs(self, subs):
        assert isinstance(subs, tuple)
        # Eagerly rename variables.
        rename = {k: v.name for k, v in subs if isinstance(v, Variable)}
        if not rename:
            return None
        step_names = frozenset((k, rename.get(v, v))
                               for k, v in self.step_names.items())
        result = MarkovProduct(self.sum_op, self.prod_op,
                               self.trans, self.time, self.step, step_names)
        lazy = tuple((k, v) for k, v in subs if not isinstance(v, Variable))
        if lazy:
            result = Subs(result, lazy)
        return result


@quote.register(MarkovProduct)
def _(arg, indent, out):
    line = "{}({}, {},".format(type(arg).__name__, repr(arg.sum_op), repr(arg.prod_op))
    out.append((indent, line))
    for value in arg._ast_values[2:]:
        quote.inplace(value, indent + 1, out)
        i, line = out[-1]
        out[-1] = i, line + ","
    i, line = out[-1]
    out[-1] = i, line[:-1] + ")"


@eager.register(MarkovProduct, AssociativeOp, AssociativeOp,
                Funsor, Variable, frozenset, frozenset)
def eager_markov_product(sum_op, prod_op, trans, time, step, step_names):
    if step:
        result = sequential_sum_product(sum_op, prod_op, trans, time, dict(step))
    elif time.name in trans.inputs:
        result = trans.reduce(prod_op, time.name)
    elif prod_op is ops.add:
        result = trans * time.size
    elif prod_op is ops.mul:
        result = trans ** time.size
    else:
        raise NotImplementedError('https://github.com/pyro-ppl/funsor/issues/233')

    return Subs(result, step_names)
