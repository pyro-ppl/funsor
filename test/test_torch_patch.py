import pytest
import torch

import funsor.torch_patch  # noqa F401
from funsor.testing import assert_close


def _cholesky_inverse(u):
    shape = u.shape
    return torch.stack([
        part.cholesky_inverse()
        for part in u.reshape((-1,) + u.shape[-2:])
    ]).reshape(shape)


@pytest.mark.parametrize("size", [1, 2, 3], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (5,), (2, 3)], ids=str)
def test_cholesky_inverse(batch_shape, size):
    x = torch.randn(batch_shape + (size, size))
    x = x.transpose(-1, -2).matmul(x)
    x = x.cholesky()
    assert_close(x.cholesky_inverse(), _cholesky_inverse(x))
