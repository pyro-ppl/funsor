from __future__ import absolute_import, division, print_function

import argparse

import torch

import funsor
import funsor.distributions as dist
import funsor.minipyro as pyro
import funsor.ops as ops
from funsor.domains import reals


@funsor.torch.function(reals(), reals())
def nonlinearity(x):
    return torch.sigmoid(x)


# a linear-Gaussian HMM
def model1(data):

    trans_noise = pyro.param(name="trans_noise")
    emit_noise = pyro.param(name="emit_noise")

    x_curr = 0.
    for t, y in enumerate(data):
        x_prev = x_curr

        # a sample statement
        x_curr = pyro.sample(
            dist.Normal(loc=x_prev, scale=trans_noise),
            name='x_{}'.format(t))

        # an observe statement
        pyro.sample(
            dist.Normal(loc=x_curr, scale=emit_noise),
            obs=y,
            name='y_{}'.format(t))

    return x_curr


# an affine linear-Gaussian HMM
def model2(data):

    trans_noise = pyro.param(name="trans_noise")
    emit_noise = pyro.param(name="emit_noise")
    trans_loc = 2.5

    x_curr = 0.
    for t, y in enumerate(data):
        x_prev = x_curr

        # a sample statement
        logits_curr = pyro.sample(
            dist.Normal(loc=x_prev, scale=1.),
            name='x_{}'.format(t))

        # an affine transform
        x_curr = trans_noise * logits_curr + trans_loc

        # an observe statement
        pyro.sample(
            dist.Normal(loc=x_curr, scale=emit_noise),
            obs=y,
            name='y_{}'.format(t))

    return x_curr


# a nonlinear-Gaussian HMM
def model3(data):

    trans_noise = pyro.param(name="trans_noise")
    emit_noise = pyro.param(name="emit_noise")

    x_curr = 0.
    for t, y in enumerate(data):
        x_prev = x_curr

        # a sample statement
        logits_curr = pyro.sample(
            dist.Normal(loc=x_prev, scale=trans_noise),
            name='x_{}'.format(t))

        # a nonlinearity to force a sample
        x_curr = nonlinearity(logits_curr)

        # an observe statement
        pyro.sample(
            dist.Normal(loc=x_curr, scale=emit_noise),
            obs=y,
            name='y_{}'.format(t))

    return x_curr


# a factorial affine/nonlinear-Gaussian HMM
def model4(data):

    trans_noise = pyro.param(name="trans_noise")
    emit_noise = pyro.param(name="emit_noise")
    trans_loc = 2.5

    x1_curr, x2_curr = 0., 0.
    for t, y in enumerate(data):
        x1_prev, x2_prev = x1_curr, x2_curr

        # a sample statement
        logits1_curr = pyro.sample(
            dist.Normal(loc=x1_prev, scale=1.),
            name='x1_{}'.format(t))

        # a nonlinearity to force a sample
        x1_curr = nonlinearity(logits1_curr)

        # a sample statement
        logits2_curr = pyro.sample(
            dist.Normal(loc=x2_prev, scale=1.),
            name='x2_{}'.format(t))

        # an affine transform
        x2_curr = trans_noise * logits2_curr + trans_loc

        # an observe statement
        pyro.sample(
            dist.Normal(loc=x1_curr + x2_curr, scale=emit_noise),
            obs=y,
            name='y_{}'.format(t))

    return x1_curr


# a 2-layer affine/nonlinear-Gaussian HMM
def model5(data):

    trans_noise = pyro.param(name="trans_noise")
    emit_noise = pyro.param(name="emit_noise")
    trans_loc = 2.5

    x1_curr, x2_curr = 0., 0.
    for t, y in enumerate(data):
        x1_prev, x2_prev = x1_curr, x2_curr

        # a sample statement
        logits1_curr = pyro.sample(
            dist.Normal(loc=x1_prev, scale=1.),
            name='x1_{}'.format(t))

        # an affine transform
        x1_curr = trans_noise * logits1_curr + trans_loc

        # a sample statement
        logits2_curr = pyro.sample(
            dist.Normal(loc=x2_prev + x1_curr, scale=1.),
            name='x2_{}'.format(t))

        # a nonlinearity to force a sample
        x2_curr = nonlinearity(logits2_curr)

        # an observe statement
        pyro.sample(
            dist.Normal(loc=x2_curr, scale=emit_noise),
            obs=y,
            name='y_{}'.format(t))

    return x2_curr


def main(args):
    """
    minipyro version of Gaussian HMM example
    """
    model = [model1, model2, model3, model4, model5][args.model - 1]

    trans_noise = pyro.param(torch.tensor(0.1, requires_grad=True), name="trans_noise")  # noqa: F841
    emit_noise = pyro.param(torch.tensor(0.5, requires_grad=True), name="emit_noise")  # noqa: F841
    data = torch.randn(args.time_steps)

    params = [node["value"] for node in pyro.trace(model).get_trace(data).values()
              if node["type"] == "param"]

    # training loop
    optim = torch.optim.Adam(params, lr=args.learning_rate)
    for step in range(args.train_steps):
        optim.zero_grad()

        tr = pyro.trace(pyro.deferred(model)).get_trace(data)

        log_prob = sum(node["fn"](node["value"])
                       for node in tr.values()
                       if node["type"] == "sample")

        # integrate out deferred variables
        log_prob = log_prob.reduce(ops.logaddexp)

        loss = -funsor.eval(log_prob)  # does all the work

        if step % 10 == 0:
            print('step {} loss = {}'.format(step, loss.item()))
        loss.backward()
        optim.step()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SMC example")
    parser.add_argument("-m", "--model", default=1, type=int)
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
