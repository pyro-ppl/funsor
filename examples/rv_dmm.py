from __future__ import absolute_import, division, print_function

import argparse

import torch

import funsor
import funsor.distributions as dist
import funsor.minipyro as pyro
import funsor.ops as ops
import funsor.rvs as rvs
from funsor.domains import reals
from funsor.expectation import Expectation
from funsor.product import Product


@funsor.torch.function(reals(), reals())
def nonlinearity(x):
    return torch.sigmoid(x)


# a nonlinear-Gaussian HMM log-joint
def model(data):

    trans_noise = pyro.param("trans_noise")
    emit_noise = pyro.param("emit_noise")
    trans_loc = 2.5

    log_prob = 0.
    x_curr = 0.
    for t, y in enumerate(data):
        x_prev = x_curr

        # a sample statement
        logits_curr = Variable("x_{}".format(t))
        log_prob += dist.Normal(loc=x_prev, scale=1.)(value=logits_curr)

        # affine transform
        logits_curr = trans_noise * logits_curr + trans_loc

        # a nonlinearity to force a sample
        x_curr = nonlinearity(logits_curr)

        # an observe statement
        log_prob += dist.Normal(loc=x_curr, scale=emit_noise)(value=y)

    return log_prob


def guide(data):
    suff_stat = nonlinearity(sum(data) / len(data))  # use sum or RNN or something...

    trans_noise = pyro.param("trans_noise_guide")
    emit_noise = pyro.param("emit_noise_guide")
    trans_loc = 2.3

    x_curr = 0.
    for t, y in enumerate(reversed(data)):  # reversed(enumerate(data))??
        x_prev = x_curr
        raw_curr = rvs.NormalRV(loc=x_curr, scale=1.)(omega="omega_{}".format(t))
        logits_curr = trans_noise * raw_curr + trans_loc
        x_curr = nonlinearity(logits_curr)
        xs["x_{}".format(t)] = raw_curr

    return Product(xs)


def main(args):
    """
    RandomVariable version of Gaussian HMM example
    """

    trans_noise = pyro.param("trans_noise", torch.tensor(0.3))  # noqa: F841
    emit_noise = pyro.param("emit_noise", torch.tensor(0.2))  # noqa: F841
    trans_noise_guide = pyro.param("trans_noise_guide", torch.tensor(0.1))  # noqa: F841
    emit_noise_guide = pyro.param("emit_noise_guide", torch.tensor(0.5))  # noqa: F841

    data = torch.randn(args.time_steps)

    params = [node["value"] for node in pyro.trace(model).get_trace(data).values()
              if node["type"] == "param"]

    # training loop
    optim = torch.optim.Adam(params, lr=args.learning_rate)
    for step in range(args.train_steps):
        optim.zero_grad()

        # compute elbo components
        guide_dist = guide(data)
        guide_log_joint = make_log_joint(guide)(data)  # TODO implement make_log_joint
        model_log_joint = model(data)

        # integrate out deferred variables
        elbo = Expectation(guide_dist, model_log_joint - guide_log_joint)
        loss = -funsor.eval(elbo)  # does all the work

        if step % 10 == 0:
            print('step {} loss = {}'.format(step, loss.item()))
        loss.backward()
        optim.step()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="RandomVariable example")
    parser.add_argument("-t", "--time-steps", default=10, type=int)
    parser.add_argument("-n", "--train-steps", default=101, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("--eager", action='store_true')
    parser.add_argument("--filter", action='store_true')
    parser.add_argument("--xfail-if-not-implemented", action='store_true')
    args = parser.parse_args()

    if args.xfail_if_not_implemented:
        try:
            main(args)
        except NotImplementedError:
            print('XFAIL')
    else:
        main(args)
