from __future__ import absolute_import, division, print_function

from collections import defaultdict, OrderedDict
from six.moves import reduce

from funsor.terms import Funsor


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


def partial_sum_product(sum_op, prod_op, factors, sum_vars=frozenset(), prod_vars=frozenset()):
    """
    Performs partial sum-product contraction of a collection of factors.

    :return: a list of partially contracted Funsors.
    :rtype: list
    """
    assert callable(sum_op)
    assert callable(prod_op)
    assert isinstance(factors, (tuple, list))
    assert all(isinstance(f, Funsor) for f in factors)
    assert isinstance(sum_vars, frozenset)
    assert isinstance(prod_vars, frozenset)

    var_to_ordinal = {}
    ordinal_to_factors = defaultdict(list)
    for f in factors:
        ordinal = prod_vars.intersection(f.inputs)
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
                results.append(f.reduce(prod_op, leaf))
            else:
                new_plates = frozenset().union(
                    *(var_to_ordinal[v] for v in remaining_sum_vars))
                if new_plates == leaf:
                    raise ValueError("intractable!")
                f = f.reduce(prod_op, leaf - new_plates)
                ordinal_to_factors[new_plates].append(f)

    return results


def sum_product(sum_op, prod_op, factors, sum_vars=frozenset(), prod_vars=frozenset()):
    """
    Performs sum-product contraction of a collection of factors.

    :return: a single contracted Funsor.
    :rtype: :class:`~funsor.terms.Funsor`
    """
    factors = partial_sum_product(sum_op, prod_op, factors, sum_vars, prod_vars)
    return reduce(prod_op, factors)
