from __future__ import absolute_import, division, print_function

from collections import defaultdict, OrderedDict
from six.moves import reduce

import funsor.ops as ops
from funsor.terms import Binary, Funsor


def naive_einsum(eqn, *terms, **kwargs):
    backend = kwargs.pop('backend', 'torch')
    if backend == 'torch':
        sum_op, prod_op = ops.add, ops.mul
    elif backend == 'pyro.ops.einsum.torch_log':
        sum_op, prod_op = ops.logaddexp, ops.add
    else:
        raise ValueError("{} backend not implemented".format(backend))

    assert isinstance(eqn, str)
    assert all(isinstance(term, Funsor) for term in terms)
    inputs, output = eqn.split('->')
    assert len(output.split(',')) == 1
    input_dims = frozenset(d for inp in inputs.split(',') for d in inp)
    output_dims = frozenset(d for d in output)
    reduce_dims = tuple(d for d in input_dims - output_dims)
    prod = terms[0]
    for term in terms[1:]:
        prod = Binary(prod_op, prod, term)
    for reduce_dim in reduce_dims:
        prod = prod.reduce(sum_op, reduce_dim)
    return prod


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


def naive_plated_einsum(eqn, *terms, **kwargs):
    backend = kwargs.pop('backend', 'torch')
    plates = kwargs.pop('plates', '')
    if backend == 'torch':
        sum_op, prod_op = ops.add, ops.mul
    elif backend == 'pyro.ops.einsum.torch_log':
        sum_op, prod_op = ops.logaddexp, ops.add
    else:
        raise ValueError("{} backend not implemented".format(backend))

    assert isinstance(eqn, str)
    assert all(isinstance(term, Funsor) for term in terms)
    inputs, output = eqn.split('->')
    assert len(output.split(',')) == 1
    input_dims = frozenset(d for inp in inputs.split(',') for d in inp)
    output_dims = frozenset(d for d in output)
    plate_dims = frozenset(plates) - output_dims
    reduce_vars = input_dims - output_dims - frozenset(plates)

    if output_dims:
        raise NotImplementedError("TODO")

    var_tree = {}
    term_tree = defaultdict(list)
    for term in terms:
        ordinal = frozenset(term.inputs) & plate_dims
        term_tree[ordinal].append(term)
        for var in term.inputs:
            if var not in plate_dims:
                var_tree[var] = var_tree.get(var, ordinal) & ordinal

    ordinal_to_var = defaultdict(set)
    for var, ordinal in var_tree.items():
        ordinal_to_var[ordinal].add(var)

    scalars = []
    while term_tree:
        leaf = max(term_tree, key=len)
        leaf_terms = term_tree.pop(leaf)
        leaf_reduce_vars = ordinal_to_var[leaf]
        for (group_terms, group_vars) in _partition(leaf_terms, leaf_reduce_vars):
            term = reduce(prod_op, group_terms).reduce(sum_op, group_vars)
            remaining_vars = frozenset(term.inputs) & reduce_vars
            if not remaining_vars:
                scalars.append(term.reduce(prod_op, leaf))
            else:
                new_plates = frozenset().union(
                    *(var_tree[v] for v in remaining_vars))
                if new_plates == leaf:
                    raise ValueError("intractable!")
                term = term.reduce(prod_op, leaf - new_plates)
                term_tree[new_plates].append(term)

    return reduce(prod_op, scalars)
