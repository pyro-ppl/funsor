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


def _unroll_plate(factors, var_to_ordinal, sum_vars, plate, step):
    # size of the plate
    size = next(iter(f.inputs[plate].size for f in factors if plate in f.inputs))
    # history of the plate
    history = 1 if step else 0

    # replicated variables
    plate_vars = set()
    for var, ordinal in var_to_ordinal.items():
        if plate in ordinal:
            plate_vars.add(var)

    # make sure that all vars in the plate are being unrolled
    assert plate_vars.issubset(sum_vars)

    # unroll variables
    for var in plate_vars:
        sum_vars -= frozenset({var})
        if var in step.keys():
            new_var = frozenset({"{}_{}".format(var.split("_")[0], i)
                                 for i in range(size)})
        elif var in step.values():
            new_var = frozenset({"{}_{}".format(var.split("_")[0], i+history)
                                 for i in range(size)})
        else:
            new_var = frozenset({"{}_{}".format(var, i+history)
                                 for i in range(size)})
        sum_vars |= new_var
        ordinal = var_to_ordinal.pop(var)
        new_ordinal = ordinal.difference({plate})
        var_to_ordinal.update({v: new_ordinal for v in new_var})

    # unroll factors
    unrolled_factors = []
    for factor in factors:
        if plate in factor.inputs:
            f_vars = plate_vars.intersection(factor.inputs)
            prev_to_var = {key: key.split("_")[0] for key in step.keys()}
            curr_to_var = {value: value.split("_")[0] for value in step.values()}
            nonmarkov_vars = f_vars - set(step.keys()) - set(step.values())
            unrolled_factors.extend([factor(
                    **{plate: i},
                    **{var: "{}_{}".format(var, i+history) for var in nonmarkov_vars},
                    **{curr: "{}_{}".format(var, i+history) for curr, var in curr_to_var.items()},
                    **{prev: "{}_{}".format(var, i) for prev, var in prev_to_var.items()},
                ) for i in range(size)])
        else:
            unrolled_factors.append(factor)

    return unrolled_factors, var_to_ordinal, sum_vars


def partial_unroll(factors, eliminate=frozenset(), plate_to_step=dict()):
    """
    Performs partial unrolling of plated factor graphs to standard factor graphs.
    Only plates with history={0, 1} are supported.

    For plates (history=0) unrolling operation appends ``_{i}`` suffix
    to variable names for index ``i`` in the plate (e.g., "x"->"x_0" for i=0).
    For markov dimensions (history=1) unrolling operation renames the suffixes
    ``var_prev`` to ``var_{i}`` and ``var_curr`` to ``var_{i+1}`` for index ``i``
    (e.g., "x_prev"->"x_0" and "x_curr"->"x_1" for i=0).
    Markov vars are assumed to have names that follow ``var_suffix`` formatting
    and specifically ``var_0`` for the initial factor (e.g.,
    ``("x_0", "x_prev", "x_curr")`` for history=1).

    :param factors: A collection of funsors.
    :type factors: tuple or list
    :param frozenset eliminate: A set of free variables to unroll,
        including both sum variables and product variable.
    :param dict plate_to_step: A dict mapping markov dimensions to
        ``step`` collections that contain ordered sequences of Markov variable names
        (e.g., ``{"time": frozenset({("x_0", "x_prev", "x_curr")})}``).
        Plates are passed with an empty ``step``.
    :return: a list of partially unrolled Funsors,
        a frozenset of partially unrolled variable names,
        and a frozenset of remaining plates.
    """
    assert isinstance(factors, (tuple, list))
    assert all(isinstance(f, Funsor) for f in factors)
    assert isinstance(eliminate, frozenset)
    assert isinstance(plate_to_step, dict)
    assert all(len(set(var.split("_")[0] for var in chain)) == 1
               and chain[0].endswith("_0")
               for step in plate_to_step.values() if step
               for chain in step)
    # process plate_to_step
    plate_to_step = plate_to_step.copy()
    for key, step in plate_to_step.items():
        # make a dict step e.g. {"x_prev": "x_curr"}; specific to history = 1
        plate_to_step[key] = {s[1]: s[2] for s in step}

    plates = frozenset(plate_to_step.keys())
    sum_vars = eliminate - plates
    unrolled_plates = {k: v for (k, v) in plate_to_step.items() if k in eliminate}
    remaining_plates = {k: v for (k, v) in plate_to_step.items() if k not in eliminate}

    var_to_ordinal = {}
    for f in factors:
        ordinal = plates.intersection(f.inputs)
        for var in set(f.inputs) - plates:
            var_to_ordinal[var] = var_to_ordinal.get(var, ordinal) & ordinal

    # first unroll plates with history=1 and highest ordinal
    # then unroll plates with history=0
    plate_to_order = {}
    for plate, step in unrolled_plates.items():
        if step:
            plate_to_order[plate] = max(len(var_to_ordinal[s]) for s in step)
        else:
            plate_to_order[plate] = 0

    # unroll one plate at a time
    for plate in sorted(unrolled_plates.keys(), key=lambda p: plate_to_order[p], reverse=True):
        step = unrolled_plates[plate]
        factors, var_to_ordinal, sum_vars = \
            _unroll_plate(factors, var_to_ordinal, sum_vars, plate, step)

    return factors, sum_vars, remaining_plates


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
    in addition to plate dimensions. Markov dimensions in transition factors
    are eliminated efficiently using the parallel-scan algorithm in
    :func:`funsor.sum_product.sequential_sum_product`. The resulting factors are then
    combined with the initial factors and final states are eliminated. Therefore,
    when Markov dimension is eliminated ``factors`` has to contain a pairs of
    initial factors and transition factors.

    :param ~funsor.ops.AssociativeOp sum_op: A semiring sum operation.
    :param ~funsor.ops.AssociativeOp prod_op: A semiring product operation.
    :param factors: A collection of funsors.
    :type factors: tuple or list
    :param frozenset eliminate: A set of free variables to eliminate,
        including both sum variables and product variable.
    :param dict plate_to_step: A dict mapping markov dimensions to
        ``step`` collections that contain ordered sequences of Markov variable names
        (e.g., ``{"time": frozenset({("x_0", "x_prev", "x_curr")})}``).
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
    # process plate_to_step
    plate_to_step = plate_to_step.copy()
    prev_to_init = {}
    for key, step in plate_to_step.items():
        # map prev to init; works for any history > 0
        for chain in step:
            init, prev = chain[:len(chain)//2], chain[len(chain)//2:-1]
            prev_to_init.update(zip(prev, init))
        # convert step to dict type required for MarkovProduct
        plate_to_step[key] = {chain[1]: chain[2] for chain in step}

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
                f = f.reduce(sum_op, frozenset(group_step.values()))
                f = f(**prev_to_init)

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
    drop = frozenset(Variable(v, trans.inputs[k])
                     for k, v in curr_to_drop.items())

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


def _get_shift(name):
    """helper function used internally in sarkka_bilmes_product"""
    return len(re.search(r"^(_PREV_)*", name).group(0)) // 6


def _shift_name(name, t):
    """helper function used internally in sarkka_bilmes_product"""
    if t >= 0:
        return t * "_PREV_" + name
    return name.replace("_PREV_" * -t, "", 1)


def _shift_funsor(f, t, global_vars):
    """helper function used internally in sarkka_bilmes_product"""
    if t == 0:
        return f
    return f(**{name: _shift_name(name, t) for name in f.inputs if name not in global_vars})


def naive_sarkka_bilmes_product(sum_op, prod_op, trans, time_var, global_vars=frozenset()):

    assert isinstance(global_vars, frozenset)

    time = time_var.name
    global_vars |= {time}

    lags = {_get_shift(name) for name in trans.inputs if name != time}
    lags.discard(0)
    if not lags:
        return naive_sequential_sum_product(sum_op, prod_op, trans, time_var, {})

    original_names = frozenset(name for name in trans.inputs
                               if name not in global_vars and not name.startswith("_PREV_"))

    duration = trans.inputs[time].size

    result = trans(**{time: duration - 1})
    for t in range(duration - 2, -1, -1):
        result = prod_op(_shift_funsor(trans(**{time: t}), duration - t - 1, global_vars), result)
        sum_vars = frozenset(_shift_name(name, duration - t - 1) for name in original_names)
        result = result.reduce(sum_op, sum_vars)

    result = result(**{name: _shift_name(name, -duration + 1) for name in result.inputs})
    return result


def sarkka_bilmes_product(sum_op, prod_op, trans, time_var, global_vars=frozenset(), num_periods=1):

    assert isinstance(global_vars, frozenset)

    time = time_var.name
    global_vars |= {time}

    lags = {_get_shift(name) for name in trans.inputs if name != time}
    lags.discard(0)
    if not lags:
        return sequential_sum_product(sum_op, prod_op, trans, time_var, {})

    period = int(reduce(lambda a, b: a * b // gcd(a, b), list(lags)))
    original_names = frozenset(name for name in trans.inputs
                               if name not in global_vars and not name.startswith("_PREV_"))
    renamed_factors = []
    duration = trans.inputs[time].size
    if duration % period != 0:
        remaining_duration = duration % period
        truncated_duration = duration - remaining_duration
        if truncated_duration == 0:
            result = trans(**{time: remaining_duration - 1})
            remaining_duration -= 1
        else:
            # chop off the rightmost set of complete chunks from trans,
            # then recursively call sarkka_bilmes_product on truncated factor
            result = sarkka_bilmes_product(
                sum_op, prod_op,
                trans(**{time: Slice(time, remaining_duration, duration, 1, duration)}),
                Variable(time, Bint[truncated_duration]),
                global_vars - {time}, num_periods
            )

        # sequentially combine remaining pieces with result
        for t in reversed(range(remaining_duration)):
            result = prod_op(_shift_funsor(trans(**{time: t}), remaining_duration - t, global_vars), result)
            sum_vars = frozenset(_shift_name(name, remaining_duration - t) for name in original_names)
            result = result.reduce(sum_op, sum_vars)

        result = result(**{name: _shift_name(name, -remaining_duration) for name in result.inputs})
        return result

    for t in range(period):
        slice_t = Slice(time, t, duration - period + t + 1, period, duration)
        factor = _shift_funsor(trans, period - t - 1, global_vars)
        factor = factor(**{time: slice_t})
        renamed_factors.append(factor)

    block_trans = reduce(prod_op, renamed_factors)
    block_step = {_shift_name(name, period): name for name in block_trans.inputs
                  if name not in global_vars and _get_shift(name) < period}
    block_time_var = Variable(time_var.name, Bint[duration // period])
    final_chunk = mixed_sequential_sum_product(
        sum_op, prod_op, block_trans, block_time_var, block_step,
        num_segments=max(1, duration // (period * num_periods)))
    final_sum_vars = frozenset(
        _shift_name(name, t) for name in original_names for t in range(1, period))
    result = final_chunk.reduce(sum_op, final_sum_vars)
    result = result(**{name: _shift_name(name, -period + 1) for name in result.inputs})
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
        bound = {k: trans.inputs[k] for k in step_names}
        bound[time.name] = time.output
        super().__init__(inputs, output, fresh, bound)
        self.sum_op = sum_op
        self.prod_op = prod_op
        self.trans = trans
        self.time = time
        self.step = step
        self.step_names = step_names

    def _alpha_convert(self, alpha_subs):
        assert set(alpha_subs).issubset(self.bound)
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
