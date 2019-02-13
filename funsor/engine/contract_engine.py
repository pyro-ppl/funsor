from __future__ import absolute_import, division, print_function

from collections import Counter

import opt_einsum

import funsor.ops as ops
from funsor.terms import Binary, Funsor, Reduction, Tensor

#####################################################
# old basic engine implementation, useful for testing
#####################################################


def _parse_reduction(op, x):
    if isinstance(x, Reduction) and x.op is op:
        yield x.arg, x.reduce_dims


def _parse_commutative(op, x):
    pending = [x]
    terms = []
    while pending:
        x = pending.pop()
        if isinstance(x, Binary) and x.op is op:
            pending.append(x.lhs)
            pending.append(x.rhs)
        else:
            terms.append(x)
    return terms


def _parse_tensors(operands):
    if all(isinstance(x, Tensor) for x in operands):
        yield operands


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
        x = x.materialize()
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

    # Apply binary contractions in order given by path.
    reduce_dim_counter = Counter(output)
    for input_ in inputs:
        reduce_dim_counter.update(input_)
    operands = list(operands)
    for x_pos, y_pos in path:
        x = operands[x_pos]
        y = operands.pop(y_pos)
        print('DEBUG x = {}, y = {}'.format(x, y))
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

        # Decide which ops to use.
        x_rdims = rdims - frozenset(y.dims)
        y_rdims = rdims - frozenset(x.dims)
        xy_rdims = rdims - x_rdims - y_rdims
        if x_rdims:
            x = x.reduce(sum_op, x_rdims)
        if y_rdims:
            y = y.reduce(sum_op, y_rdims)
        if xy_rdims:
            xy = x.contract(sum_op, prod_op, y, xy_rdims)
            if isinstance(xy, Reduction):
                xy = y.contract(sum_op, prod_op, x, xy_rdims)
        else:
            xy = x.binary(prod_op, y)

        operands[x_pos] = xy

    # Apply an optional final reduction.
    # This should only happen when len(operands) == 1 on entry.
    x, = operands
    rdims = frozenset(x.dims) - frozenset(reduce_dim_counter)
    if rdims:
        x = x.reduce(sum_op, rdims)
    assert frozenset(x.dims) == frozenset(reduce_dim_counter)
    return x


def eval(x):
    """original contract-based eval implementation, useful for testing"""
    # Handle trivial case
    if isinstance(x, Tensor):
        return x.materialize()

    # Handle sum-product contractions.
    for arg, reduce_dims in _parse_reduction(ops.add, x):
        operands = _parse_commutative(ops.mul, arg)
        operands = tuple(x.materialize() for x in operands)
        for tensors in _parse_tensors(operands):
            dims = tuple(d for d in arg.dims if d not in reduce_dims)
            return _contract_tensors(*tensors, dims=dims)
        return _contract(ops.add, ops.mul, operands, reduce_dims)

    # Handle log-sum-product-exp contractions.
    for arg, reduce_dims in _parse_reduction(ops.logaddexp, x):
        operands = _parse_commutative(ops.add, arg)
        operands = tuple(x.materialize() for x in operands)
        for tensors in _parse_tensors(operands):
            dims = tuple(d for d in arg.dims if d not in reduce_dims)
            return _contract_tensors(*tensors, dims=dims, backend='pyro.ops.einsum.torch_log')
        return _contract(ops.add, ops.mul, operands, reduce_dims)


__all__ = [
    'eval',
]
