from __future__ import absolute_import, division, print_function

import pytest

import torch
import funsor


@pytest.mark.parametrize('materialize_f', [False, True])
@pytest.mark.parametrize('materialize_g', [False, True])
def test_mm(materialize_f, materialize_g):

    @funsor.of_shape(3, 4)
    def f(i, j):
        return i + j

    if materialize_f:
        f = f.materialize()

    @funsor.of_shape(4, 5)
    def g(j, k):
        return j + k

    if materialize_f:
        f = f.materialize()

    h = (f * g).sum('j')
    eval_h = funsor.eval(h)
    assert isinstance(eval_h, funsor.Tensor)
    assert eval_h.dims == h.dims
    assert eval_h.shape == h.shape
    for i in range(3):
        for k in range(5):
            assert eval_h[i, k] == h[i, k].materialize()


@pytest.mark.parametrize('materialize_f', [False, True])
@pytest.mark.parametrize('materialize_g', [False, True])
def test_logsumproductexp(materialize_f, materialize_g):

    @funsor.of_shape(3, 4)
    def f(i, j):
        return i + j

    if materialize_f:
        f = f.materialize()

    @funsor.of_shape(4, 5)
    def g(j, k):
        return j + k

    if materialize_f:
        f = f.materialize()

    log_prob = funsor.Tensor(('log_prob',), torch.randn(10))
    h = (log_prob[f] + log_prob[g]).logsumexp('j')

    eval_h = funsor.eval(h)
    assert isinstance(eval_h, funsor.Tensor)
    assert eval_h.dims == h.dims
    assert eval_h.shape == h.shape
    for i in range(3):
        for k in range(5):
            assert (eval_h[i, k] - h[i, k].materialize()) < 1e-6
