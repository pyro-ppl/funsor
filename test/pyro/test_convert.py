from __future__ import absolute_import, division, print_function

import pyro.distributions as dist
import pytest
import torch

from funsor.pyro.convert import dist_to_funsor, funsor_to_tensor, tensor_to_funsor
from funsor.testing import assert_close
from funsor.torch import Tensor

EVENT_SHAPES = [(), (1,), (5,), (4, 3)]
BATCH_SHAPES = [(), (1,), (4,), (2, 3), (1, 2, 1, 3, 1)]


@pytest.mark.parametrize("event_shape", EVENT_SHAPES, ids=str)
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
def test_tensor_funsor_tensor(batch_shape, event_shape):
    event_dim = len(event_shape)
    t = torch.randn(batch_shape + event_shape)
    f = tensor_to_funsor(t, event_dim=event_dim)
    t2 = funsor_to_tensor(f, ndims=t.dim())
    assert_close(t2, t)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
@pytest.mark.parametrize("cardinality", [2, 3, 5])
def test_dist_to_funsor_categorical(batch_shape, cardinality):
    logits = torch.randn(batch_shape + (cardinality,))
    logits -= logits.logsumexp(dim=-1, keepdim=True)
    d = dist.Categorical(logits=logits)
    f = dist_to_funsor(d)
    assert isinstance(f, Tensor)
    expected = tensor_to_funsor(logits, event_dim=1)
    assert_close(f, expected)
