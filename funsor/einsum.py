from __future__ import absolute_import, division, print_function

from six.moves import reduce

import funsor.ops as ops
from funsor.interpreter import interpretation, reinterpret
from funsor.optimizer import apply_optimizer
from funsor.terms import Funsor, reflect
from funsor.contract import sum_product


def naive_einsum(eqn, *terms, **kwargs):
    backend = kwargs.pop('backend', 'torch')
    if backend == 'torch':
        sum_op, prod_op = ops.add, ops.mul
    elif backend in ('pyro.ops.einsum.torch_log', 'pyro.ops.einsum.torch_marginal'):
        sum_op, prod_op = ops.logaddexp, ops.add
    else:
        raise ValueError("{} backend not implemented".format(backend))

    assert isinstance(eqn, str)
    assert all(isinstance(term, Funsor) for term in terms)
    inputs, output = eqn.split('->')
    assert len(output.split(',')) == 1
    input_dims = frozenset(d for inp in inputs.split(',') for d in inp)
    output_dims = frozenset(output)
    reduce_dims = input_dims - output_dims
    return reduce(prod_op, terms).reduce(sum_op, reduce_dims)


def naive_plated_einsum(eqn, *terms, **kwargs):
    """
    Implements Tensor Variable Elimination (Algorithm 1 in [Obermeyer et al 2019])

    [Obermeyer et al 2019] Obermeyer, F., Bingham, E., Jankowiak, M., Chiu, J.,
        Pradhan, N., Rush, A., and Goodman, N.  Tensor Variable Elimination for
        Plated Factor Graphs, 2019
    """
    plates = kwargs.pop('plates', '')
    if not plates:
        return naive_einsum(eqn, *terms, **kwargs)

    backend = kwargs.pop('backend', 'torch')
    if backend == 'torch':
        sum_op, prod_op = ops.add, ops.mul
    elif backend in ('pyro.ops.einsum.torch_log', 'pyro.ops.einsum.torch_marginal'):
        sum_op, prod_op = ops.logaddexp, ops.add
    else:
        raise ValueError("{} backend not implemented".format(backend))

    assert isinstance(eqn, str)
    assert all(isinstance(term, Funsor) for term in terms)
    inputs, output = eqn.split('->')
    inputs = inputs.split(',')
    assert len(inputs) == len(terms)
    assert len(output.split(',')) == 1
    input_dims = frozenset(d for inp in inputs for d in inp)
    output_dims = frozenset(d for d in output)
    plate_dims = frozenset(plates) - output_dims
    reduce_vars = input_dims - output_dims - frozenset(plates)

    output_plates = output_dims & frozenset(plates)
    if not all(output_plates.issubset(inp) for inp in inputs):
        raise NotImplementedError("TODO")

    eliminate = plate_dims | reduce_vars
    return sum_product(sum_op, prod_op, terms, eliminate, frozenset(plates))


def einsum(eqn, *terms, **kwargs):
    with interpretation(reflect):
        naive_ast = naive_plated_einsum(eqn, *terms, **kwargs)
        optimized_ast = apply_optimizer(naive_ast)
    return reinterpret(optimized_ast)  # eager by default
