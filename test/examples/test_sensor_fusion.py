from collections import OrderedDict

import pytest
import torch

import funsor.ops as ops
from funsor.cnf import Contraction
from funsor.domains import bint, reals
from funsor.gaussian import Gaussian
from funsor.pyro.convert import dist_to_funsor, matrix_and_mvn_to_funsor
from funsor.terms import Subs, Variable
from funsor.testing import random_mvn
from funsor.torch import Tensor


@pytest.mark.xfail(reason="missing pattern")
def test_end_to_end():
    data = Tensor(torch.randn(2, 2), OrderedDict([("time", bint(2))]))

    bias_dist = dist_to_funsor(random_mvn((), 2))

    trans_mat = torch.randn(3, 3)
    trans_mvn = random_mvn((), 3)
    trans = matrix_and_mvn_to_funsor(trans_mat, trans_mvn, (), "prev", "curr")

    obs_mat = torch.randn(3, 2)
    obs_mvn = random_mvn((), 2)
    obs = matrix_and_mvn_to_funsor(obs_mat, obs_mvn, (), "state", "obs")

    log_prob = 0
    bias = Variable("bias", reals(2))
    log_prob += bias_dist(value=bias)

    state_0 = Variable("state_0", reals(3))
    log_prob += obs(state=state_0, obs=bias + data(time=0))

    state_1 = Variable("state_1", reals(3))
    log_prob += trans(prev=state_0, curr=state_1)
    log_prob += obs(state=state_1, obs=bias + data(time=1))

    log_prob = log_prob.reduce(ops.logaddexp)
    assert isinstance(log_prob, Tensor), log_prob.pretty()


@pytest.mark.xfail(reason="missing pattern")
def test_affine_subs():
    # This was recorded from test_end_to_end.
    x = Subs(
     Gaussian(
      torch.tensor([1.3027106523513794, 1.4167094230651855, -0.9750942587852478, 0.5321089029312134, -0.9039931297302246], dtype=torch.float32),  # noqa
      torch.tensor([[1.0199567079544067, 0.9840421676635742, -0.473368763923645, 0.34206756949424744, -0.7562517523765564], [0.9840421676635742, 1.511502742767334, -1.7593903541564941, 0.6647964119911194, -0.5119513273239136], [-0.4733688533306122, -1.7593903541564941, 3.2386727333068848, -0.9345928430557251, -0.1534711718559265], [0.34206756949424744, 0.6647964119911194, -0.9345928430557251, 0.3141004145145416, -0.12399007380008698], [-0.7562517523765564, -0.5119513273239136, -0.1534711718559265, -0.12399007380008698, 0.6450173854827881]], dtype=torch.float32),  # noqa
      (('state_1_b6',
        reals(3,),),
       ('obs_b2',
        reals(2,),),)),
     (('obs_b2',
       Contraction(ops.nullop, ops.add,
        frozenset(),
        (Variable('bias_b5', reals(2,)),
         Tensor(
          torch.tensor([-2.1787893772125244, 0.5684312582015991], dtype=torch.float32),  # noqa
          (),
          'real'),)),),))
    assert isinstance(x, (Gaussian, Contraction)), x.pretty()
