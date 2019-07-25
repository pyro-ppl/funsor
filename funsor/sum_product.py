from collections import OrderedDict, defaultdict
from functools import reduce

import torch

from funsor.domains import bint
from funsor.ops import UNITS, Op
from funsor.terms import Funsor, Number
from funsor.torch import Tensor, align_tensor


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


def sum_product(sum_op, prod_op, factors, eliminate=frozenset(), plates=frozenset()):
    """
    Performs sum-product contraction of a collection of factors.

    :return: a single contracted Funsor.
    :rtype: :class:`~funsor.terms.Funsor`
    """
    factors = partial_sum_product(sum_op, prod_op, factors, eliminate, plates)
    return reduce(prod_op, factors, Number(UNITS[prod_op]))


# TODO Promote this to a first class funsor and move this logic
# into eager_cat for Tensor.
def Cat(parts, name):
    if len(parts) == 1:
        return parts[0]
    if len(set(part.output for part in parts)) > 1:
        raise NotImplementedError("TODO")
    if not all(isinstance(part, Tensor) for part in parts):
        raise NotImplementedError("TODO")

    inputs = OrderedDict()
    for x in parts:
        inputs.update(x.inputs)
    tensors = []
    for part in parts:
        inputs[name] = part.inputs[name]
        shape = tuple(d.size for d in inputs.values())
        tensors.append(align_tensor(inputs, part).expand(shape))

    dim = tuple(inputs).index(name)
    tensor = torch.cat(tensors, dim=dim)
    inputs[name] = bint(tensor.size(dim))
    return Tensor(tensor, inputs, dtype=parts[0].dtype)


# TODO Promote this to a first class funsor, enabling zero-copy slicing.
def Slice(name, *args):
    start = 0
    step = 1
    bound = None
    if len(args) == 1:
        stop = args[0]
        bound = stop
    elif len(args) == 2:
        start, stop = args
        bound = stop
    elif len(args) == 3:
        start, stop, step = args
        bound = stop
    elif len(args) == 4:
        start, stop, step, bound = args
    else:
        raise ValueError
    if step <= 0:
        raise ValueError
    # FIXME triggers tensor op
    # TODO move this logic up into funsor.torch.arange?
    data = torch.arange(start, stop, step)
    inputs = OrderedDict([(name, bint(len(data)))])
    return Tensor(data, inputs, dtype=bound)


def sequential_sum_product(sum_op, prod_op, trans, time, prev, curr):
    """
    For a funsor ``trans`` with dimensions ``time``, ``prev`` and ``curr``,
    computes a recursion equivalent to::

        tail_time = 1 + arange("time", trans.inputs["time"].size - 1)
        tail = sequential_sum_product(sum_op, prod_op,
                                      trans(time=tail_time),
                                      "time", "prev", "curr")
        return prod_op(trans(time=0)(curr="drop"), tail(prev="drop")) \
           .reduce(sum_op, "drop")

    but does so efficiently in parallel in O(log(time)).
    """
    assert isinstance(sum_op, Op)
    assert isinstance(prod_op, Op)
    assert isinstance(trans, Funsor)
    assert isinstance(time, str)
    assert isinstance(prev, str)
    assert isinstance(curr, str)

    while trans.inputs["time"].size > 1:
        duration = trans.inputs["time"].size
        even_duration = duration // 2 * 2
        # TODO support syntax
        # x = trans(time=slice(0, even_duration, 2), ...)
        x = trans(time=Slice("time", 0, even_duration, 2, duration), curr="_drop")
        y = trans(time=Slice("time", 1, even_duration, 2, duration), prev="_drop")
        contracted = prod_op(x, y).reduce(sum_op, "_drop")
        if duration > even_duration:
            extra = trans(time=Slice("time", duration - 1, duration))
            contracted = Cat((contracted, extra), "time")
        trans = contracted
    return trans(time=0)
