# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import funsor.ops as ops
from funsor.einsum.util import Tensordot
from funsor.util import get_backend


def einsum(equation, *operands):
    """
    Log-sum-exp implementation of einsum.
    """
    if get_backend() != "jax":
        # NB: rename symbols to support NumPy, which allow only symbols a-z.
        symbols = sorted(set(equation) - set(',->'))
        rename = dict(zip(symbols, 'abcdefghijklmnopqrstuvwxyz'))
        equation = ''.join(rename.get(s, s) for s in equation)

    inputs, output = equation.split('->')
    if inputs == output:
        return operands[0][...]  # create a new object
    inputs = inputs.split(',')

    shifts = []
    exp_operands = []
    for dims, operand in zip(inputs, operands):
        shift = ops.detach(operand)
        for i, dim in enumerate(dims):
            if dim not in output:
                shift = ops.amax(shift, i, keepdims=True)
        # avoid nan due to -inf - -inf
        shift = ops.clamp(shift, ops.finfo(shift).min, None)
        exp_operands.append(ops.exp(operand - shift))

        # permute shift to match output
        shift = shift.reshape([size for size, dim in zip(operand.shape, dims) if dim in output])
        if len(shift.shape) > 0:
            shift = shift.reshape((1,) * (len(output) - shift.ndim) + shift.shape)
            dims = [dim for dim in dims if dim in output]
            dims = [dim for dim in output if dim not in dims] + dims
            shift = ops.permute(shift, [dims.index(dim) for dim in output])
        shifts.append(shift)

    result = ops.log(ops.einsum(equation, *exp_operands))
    return sum(shifts + [result])


tensordot = Tensordot(einsum)
transpose = ops.permute

__all__ = ["einsum", "tensordot", "transpose"]
