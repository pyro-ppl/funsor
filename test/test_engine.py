from __future__ import absolute_import, division, print_function

import pytest

import torch
import funsor

from funsor.engine import eval as main_eval
from funsor.engine.contract_engine import eval as _contract_eval
from funsor.engine.engine import EagerEval
from funsor.engine.optimizer import apply_optimizer


def unoptimized_eval(x): return EagerEval(main_eval)(x)


def optimized_eval(x): return EagerEval(main_eval)(apply_optimizer(x))


def contract_eval(x): return _contract_eval(x)  # for pytest param naming


@pytest.mark.parametrize('eval', [unoptimized_eval, optimized_eval, contract_eval])
@pytest.mark.parametrize('materialize_f', [False, True])
@pytest.mark.parametrize('materialize_g', [False, True])
def test_mm(eval, materialize_f, materialize_g):

    @funsor.of_shape(3, 4)
    def f(i, j):
        return i + j

    if materialize_f:
        f = f.materialize()

    @funsor.of_shape(4, 5)
    def g(j, k):
        return j + k

    if materialize_g:
        g = g.materialize()

    h = (f * g).sum('j')
    eval_h = eval(h)
    assert isinstance(eval_h, funsor.Tensor)
    assert eval_h.dims == h.dims
    assert eval_h.shape == h.shape
    for i in range(3):
        for k in range(5):
            assert eval_h[i, k] == h[i, k].materialize()


@pytest.mark.parametrize('eval', [unoptimized_eval, optimized_eval, contract_eval])
@pytest.mark.parametrize('materialize_f', [False, True])
@pytest.mark.parametrize('materialize_g', [False, True])
def test_logsumproductexp(eval, materialize_f, materialize_g):

    @funsor.of_shape(3, 4)
    def f(i, j):
        return i + j

    if materialize_f:
        f = f.materialize()

    @funsor.of_shape(4, 5)
    def g(j, k):
        return j + k

    if materialize_g:
        g = g.materialize()

    log_prob = funsor.Tensor(('log_prob',), torch.randn(10))
    h = (log_prob[f] + log_prob[g]).logsumexp('j')

    eval_h = eval(h)
    assert isinstance(eval_h, funsor.Tensor)
    assert eval_h.dims == h.dims
    assert eval_h.shape == h.shape
    for i in range(3):
        for k in range(5):
            assert (eval_h[i, k] - h[i, k].materialize()) < 1e-6
