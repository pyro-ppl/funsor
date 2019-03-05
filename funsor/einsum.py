from __future__ import absolute_import, division, print_function

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


def naive_plated_einsum(eqn, *terms, **kwargs):
    # FIXME not working correctly yet...
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
    plate_dims = tuple(d for d in frozenset(plates) - output_dims)
    reduce_dims = tuple(d for d in input_dims - output_dims - frozenset(plates))
    prod = terms[0]
    for term in terms[1:]:
        prod = Binary(prod_op, prod, term)
    for plate_dim in plate_dims:
        prod = prod.reduce(prod_op, plate_dim)
    for reduce_dim in reduce_dims:
        prod = prod.reduce(sum_op, reduce_dim)
    return prod
