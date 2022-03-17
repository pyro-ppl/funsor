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
    L = torch.randn(batch_shape + (n, n)).tril_()
    L.diagonal(dim1=-2, dim2=-1).abs_().add_(0.2)
    U = L.transpose(-1, -2)

    actual = ops.triangular_solve(b, L, upper=False)
    expected = torch.linalg.solve_triangular(L, b, upper=False)
    assert_close(actual, expected)
    assert_close(L @ actual, b, atol=1e-4, rtol=1e-3)

    actual = ops.triangular_solve(b, U, upper=True)
    expected = torch.linalg.solve_triangular(U, b, upper=True)
    assert_close(actual, expected)
    assert_close(U @ actual, b, atol=1e-4, rtol=1e-3)

    actual = ops.triangular_solve(b, L, upper=False, transpose=True)
    expected = torch.linalg.solve_triangular(L.transpose(-1, -2), b, upper=True)
    assert_close(actual, expected)
    assert_close(L.transpose(-1, -2) @ actual, b, atol=1e-4, rtol=1e-3)

    actual = ops.triangular_solve(b, U, upper=True, transpose=True)
    expected = torch.linalg.solve_triangular(U.transpose(-1, -2), b, upper=False)
    assert_close(actual, expected)
    assert_close(U.transpose(-1, -2) @ actual, b, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_triangular_inv(batch_shape, n):
    import torch

    L = torch.randn(batch_shape + (n, n)).tril_()
    L.diagonal(dim1=-2, dim2=-1).abs_().add_(0.2)
    U = L.transpose(-1, -2)
    eye = ops.new_eye(L, L.shape[:-1])

    actual = ops.triangular_inv(L, upper=False)
    assert_close(L @ actual, eye, atol=1e-5, rtol=None)
    assert_close(actual @ L, eye, atol=1e-5, rtol=None)

    actual = ops.triangular_inv(U, upper=True)
    assert_close(U @ actual, eye, atol=1e-5, rtol=None)
    assert_close(actual @ U, eye, atol=1e-5, rtol=None)
