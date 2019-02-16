from __future__ import absolute_import, division, print_function

from collections import Counter

import opt_einsum
from six.moves import reduce

import funsor.ops as ops
from funsor.engine import materialize
from funsor.pattern import match_commutative, try_match_reduction, try_match_tensors
from funsor.terms import Funsor
from funsor.torch import Tensor

#####################################################
# old basic engine implementation, useful for testing
#####################################################


def _contract_tensors(*operands, **kwargs):
    r"""
    Sum-product contraction operation.

    :param tuple dims: a tuple of strings of output dimensions. Any input dim
        not requested as an output dim will be summed out.
    :param \*operands: multiple :class:`Funsor`s.
    :param tuple dims: An optional tuple of output dims to preserve.
        Defaults to ``()``, meaning all dims are contracted.
    :param str backend: An opt_einsum backend, defaults to 'torch'.
    :return: A contracted funsor.
    :rtype: Funsor
    """
    assert all(isinstance(x, Funsor) for x in operands)
    dims = kwargs.pop('dims', ())
    assert isinstance(dims, tuple)
    assert all(isinstance(d, str) for d in dims)
    kwargs.setdefault('backend', 'torch')
    args = []
    for x in operands:
        x = materialize(x)
        if not isinstance(x, Tensor):
            raise NotImplementedError
        args.extend([x.data, x.dims])
    args.append(dims)
    data = opt_einsum.contract(*args, **kwargs)
    return Tensor(dims, data)


def _contract(sum_op, prod_op, operands, reduce_dims, default_cost=10):
    # Optimize operation order using opt_einsum.
    reduce_dims = frozenset(reduce_dims)
    inputs = []
    size_dict = {}
    for x in operands:
        inputs.append(x.dims)
        size_dict.update(x.schema)
    size_dict = {d: s if isinstance(s, int) else default_cost
                 for d, s in size_dict.items()}
    output = frozenset().union(*inputs) - reduce_dims
    path = opt_einsum.paths.greedy(inputs, output, size_dict)

    # Apply pairwise contractions in order given by path.
    reduce_dim_counter = Counter(output)
    for input_ in inputs:
        reduce_dim_counter.update(input_)
    operands = list(operands)
    for x_pos, y_pos in path:
        x = operands[x_pos]
        y = operands.pop(y_pos)
        x_dims = reduce_dims.intersection(x.dims)
        y_dims = reduce_dims.intersection(y.dims)
        reduce_dim_counter.subtract(x_dims)
        reduce_dim_counter.subtract(y_dims)
        rdims = []
        for d in x_dims | y_dims:
            if reduce_dim_counter[d] == 0:
                del reduce_dim_counter[d]
                rdims.append(d)
        rdims = frozenset(rdims)
        xy = _pairwise_contract(sum_op, prod_op, x, y, rdims)
        operands[x_pos] = xy
        reduce_dim_counter.update(reduce_dims.intersection(xy.dims))

    # Apply an optional final reduction.
    # This should only happen when len(operands) == 1 on entry.
    x, = operands
    rdims = frozenset(x.dims) - frozenset(output)
    if rdims:
        x = x.reduce(sum_op, rdims)
    assert frozenset(x.dims) == frozenset(output)
    return x


def _pairwise_contract(sum_op, prod_op, x, y, rdims):
    if not rdims:
        xy = x.binary(prod_op, y)
        return xy
    factors = match_commutative(prod_op, x, y)

    # Reduce one dim at a time.
    for dim in rdims:
        dim_factors = [f for f in factors if dim in f.dims]
        factors = [f for f in factors if dim not in f.dims]
        if len(dim_factors) == 1:
            xy = dim_factors[0].reduce(sum_op, frozenset([dim]))
        elif len(dim_factors) == 2:
            x, y = dim_factors
            try:
                xy = x.contract(sum_op, prod_op, y, frozenset([dim]))
            except NotImplementedError:
                xy = y.contract(sum_op, prod_op, x, frozenset([dim]))
        else:
            raise NotImplementedError
        assert dim not in xy.dims
        factors.extend(match_commutative(prod_op, xy))

    return reduce(prod_op, factors)


def eval(x):
    """original contract-based eval implementation, useful for testing"""
    assert isinstance(x, Funsor)

    # Handle sum-product contractions.
    for arg, reduce_dims in try_match_reduction(ops.add, x):
        operands = match_commutative(ops.mul, arg)
        operands = tuple(materialize(x) for x in operands)
        for tensors in try_match_tensors(operands):
            dims = tuple(d for d in arg.dims if d not in reduce_dims)
            return _contract_tensors(*tensors, dims=dims)
        return _contract(ops.add, ops.mul, operands, reduce_dims)

    # Handle log-sum-product-exp contractions.
    for arg, reduce_dims in try_match_reduction(ops.logaddexp, x):
        operands = match_commutative(ops.add, arg)
        operands = tuple(materialize(x) for x in operands)
        for tensors in try_match_tensors(operands):
            dims = tuple(d for d in arg.dims if d not in reduce_dims)
            return _contract_tensors(*tensors, dims=dims, backend='pyro.ops.einsum.torch_log')
        return _contract(ops.logaddexp, ops.add, operands, reduce_dims)

    return materialize(x)


__all__ = [
    'eval',
]
