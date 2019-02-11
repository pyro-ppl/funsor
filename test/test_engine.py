from __future__ import absolute_import, division, print_function

import pytest

import torch
import funsor

from funsor.engine import eval
from funsor.engine.contract_engine import eval as contract_eval
from funsor.engine.engine import EagerEval
from funsor.engine.optimizer import apply_optimizer


@pytest.mark.parametrize('materialize_f', [False, True])
@pytest.mark.parametrize('materialize_g', [False, True])
@pytest.mark.parametrize('optimize', [True, False])
@pytest.mark.parametrize('reflect', [False])  # TODO add check for reflect=True
def test_mm(materialize_f, materialize_g, optimize, reflect):

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
    if reflect:
        eval_h = eval(h)
        assert eval_h == h
    else:
        eval_h = EagerEval(eval)(h)
        assert isinstance(eval_h, funsor.Tensor)
        assert eval_h.dims == h.dims
        assert eval_h.shape == h.shape
        for i in range(3):
            for k in range(5):
                assert eval_h[i, k] == h[i, k].materialize()


@pytest.mark.parametrize('materialize_f', [False, True])
@pytest.mark.parametrize('materialize_g', [False, True])
@pytest.mark.parametrize('optimize', [True, False])
@pytest.mark.parametrize('reflect', [False])  # TODO add check for reflect=True
def test_logsumproductexp(materialize_f, materialize_g, optimize, reflect):

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

    if reflect:
        eval_h = eval(h)
        assert eval_h == h
    else:
        h = apply_optimizer(h) if optimize else h
        eval_h = EagerEval(eval)(h)
        assert isinstance(eval_h, funsor.Tensor)
        assert eval_h.dims == h.dims
        assert eval_h.shape == h.shape
        for i in range(3):
            for k in range(5):
                assert (eval_h[i, k] - h[i, k].materialize()) < 1e-6
