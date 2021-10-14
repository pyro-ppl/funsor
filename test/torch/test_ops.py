# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest

import funsor.ops as ops
from funsor.testing import assert_close, requires_backend

pytestmark = requires_backend("torch")


@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_triangular_solve(batch_shape, m, n):
    import torch

    b = torch.randn(batch_shape + (n, m))
    A = torch.randn(batch_shape + (n, n)).tril_()
    A.diagonal(dim1=-2, dim2=-1).abs_()

    actual = ops.triangular_solve(b, A, upper=False)
    expected = torch.triangular_solve(b, A, upper=False).solution
    assert_close(actual, expected)
