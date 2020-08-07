# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import operator

from functools import reduce

import funsor.ops as ops
from funsor.einsum.util import Tensordot, broadcast_all


def einsum(equation, *operands):
    """
    Forward-max-sum backward-argmax implementation of einsum.
    This assumes all operands have a ``._pyro_dims`` attribute set.
    """
    inputs, output = equation.split('->')
    inputs = inputs.split(',')

    contract_dims = ''.join(sorted(set().union(*inputs) - set(output)))
    dims = output + contract_dims
    result = reduce(operator.add, broadcast_all(*operands, inputs=inputs, dims=dims))
    if contract_dims:
        output_shape = result.shape[:len(output)]
        result = ops.amax(result.reshape(output_shape + (-1,)), -1)
    elif result is operands[0]:
        result = result[...]  # create a new object
    assert len(result.shape) == len(output)

    return result


tensordot = Tensordot(einsum)
transpose = ops.permute

__all__ = ["transpose", "einsum", "tensordot"]
