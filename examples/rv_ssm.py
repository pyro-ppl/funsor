from __future__ import absolute_import, division, print_function

import argparse

import torch

import funsor
import funsor.distributions as dist
import funsor.minipyro as pyro
import funsor.ops as ops
import funsor.rvs as rvs


# a linear-Gaussian HMM
def model(data):

    trans_noise = pyro.param(name="trans_noise")
    emit_noise = pyro.param(name="emit_noise")

    log_prob = 0.
    x_curr = 0.
    for t, y in enumerate(data):
        x_prev = x_curr

        # a sample statement
        x_curr = rvs.NormalRV(loc=x_prev, scale=trans_noise)(omega="omega_{}".format(t))

        # an observe statement: accumulate conditional densities
        log_prob += dist.Normal(loc=x_curr, scale=emit_noise)(value=y)

    return log_prob


def main(args):
    """
    RandomVariable version of Gaussian HMM example
    """

    trans_noise = pyro.param(torch.tensor(0.1, requires_grad=True), name="trans_noise")  # noqa: F841
    emit_noise = pyro.param(torch.tensor(0.5, requires_grad=True), name="emit_noise")  # noqa: F841
    data = torch.randn(args.time_steps)

    params = [node["value"] for node in pyro.trace(model).get_trace(data).values()
              if node["type"] == "param"]

    # training loop
    optim = torch.optim.Adam(params, lr=args.learning_rate)
    for step in range(args.train_steps):
        optim.zero_grad()

        log_prob = model(data)

        # integrate out deferred variables
        log_prob = log_prob.reduce(ops.logaddexp)

        loss = -funsor.eval(log_prob)  # does all the work

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
