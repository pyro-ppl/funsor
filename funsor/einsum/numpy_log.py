# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as np

from funsor.einsum.util import Tensordot


def einsum(equation, *operands):
    """
    Log-sum-exp implementation of einsum.
    """
    inputs, output = equation.split('->')
    if inputs == output:
        return operands[0][...]  # create a new object
    inputs = inputs.split(',')

    shifts = []
    exp_operands = []
    for dims, operand in zip(inputs, operands):
        shift = operand
        for i, dim in enumerate(dims):
            if dim not in output:
                shift = np.max(shift, i, keepdims=True)
        # avoid nan due to -inf - -inf
        shift = np.clip(shift, a_min=np.finfo(shift.dtype).min, a_max=None)
        exp_operands.append(np.exp(operand - shift))

        # permute shift to match output
        shift = shift.reshape([size for size, dim in zip(operand.shape, dims) if dim in output])
        if shift.ndim:
            shift = shift.reshape((1,) * (len(output) - shift.ndim) + shift.shape)
            dims = [dim for dim in dims if dim in output]
            dims = [dim for dim in output if dim not in dims] + dims
            shift = np.transpose(shift, [dims.index(dim) for dim in output])
        shifts.append(shift)

    result = np.log(np.einsum(equation, *exp_operands))
    return sum(shifts + [result])


tensordot = Tensordot(einsum)
transpose = np.transpose

__all__ = ["transpose", "einsum", "tensordot"]
