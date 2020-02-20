# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from funsor import ops

EINSUM_SYMBOLS_BASE = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


class Tensordot:
    """
    Creates a tensordot implementation from an einsum implementation.
    """
    def __init__(self, einsum):
        self.einsum = einsum

    # Copyright (c) 2014 Daniel Smith
    # SPDX-License-Identifier: MIT
    # This function is copied and adapted from:
    # https://github.com/dgasmith/opt_einsum/blob/a6dd686/opt_einsum/backends/torch.py
    def __call__(self, x, y, axes=2):
        xnd = len(x.shape)
        ynd = len(y.shape)

        # convert int argument to (list[int], list[int])
        if isinstance(axes, int):
            axes = range(xnd - axes, xnd), range(axes)

        # convert (int, int) to (list[int], list[int])
        if isinstance(axes[0], int):
            axes = (axes[0],), axes[1]
        if isinstance(axes[1], int):
            axes = axes[0], (axes[1],)

        # initialize empty indices
        x_ix = [None] * xnd
        y_ix = [None] * ynd
        out_ix = []

        # fill in repeated indices
        available_ix = iter(EINSUM_SYMBOLS_BASE)
        for ax1, ax2 in zip(*axes):
            repeat = next(available_ix)
            x_ix[ax1] = repeat
            y_ix[ax2] = repeat

        # fill in the rest, and maintain output order
        for i in range(xnd):
            if x_ix[i] is None:
                leave = next(available_ix)
                x_ix[i] = leave
                out_ix.append(leave)
        for i in range(ynd):
            if y_ix[i] is None:
                leave = next(available_ix)
                y_ix[i] = leave
                out_ix.append(leave)

        # form full string and contract!
        einsum_str = "{},{}->{}".format(*map("".join, (x_ix, y_ix, out_ix)))
        return self.einsum(einsum_str, x, y)


def rename_equation(equation, *operands):
    """
    Renames symbols in an einsum/ubersum equation to match the
    ``.pyro_dims`` attributes of packed ``operands``.
    """
    inputs, outputs = equation.split('->')
    inputs = inputs.split(',')
    assert len(inputs) == len(operands)
    rename = {old: new
              for input_, operand in zip(inputs, operands)
              for old, new in zip(input_, operand._pyro_dims)}
    return ''.join(rename.get(s, s) for s in equation)


class Array(np.ndarray):
    pass


def broadcast_all(*values, **kwargs):
    """
    Packed broadcasting of multiple tensors.
    """
    dims = kwargs.get('dims')
    sizes = {dim: size for value in values for dim, size in zip(value._pyro_dims, value.shape)}
    if dims is None:
        dims = ''.join(sorted(sizes))
    else:
        assert set(dims) == set(sizes)
    shape = tuple(sizes[dim] for dim in dims)
    values = list(values)
    for i, x in enumerate(values):
        old_dims = x._pyro_dims
        if old_dims != dims:
            x = ops.permute(x, tuple(old_dims.index(dim) for dim in dims if dim in old_dims))
            x = x.reshape(tuple(sizes[dim] if dim in old_dims else 1 for dim in dims))
            x = ops.expand(x, shape)
            # workaround: ndarray does not allow setting attribute "_pyro_dims"
            if isinstance(x, np.ndarray):
                x = x.view(Array)
            x._pyro_dims = dims
            assert len(x.shape) == len(x._pyro_dims)
            values[i] = x
    return tuple(values)
