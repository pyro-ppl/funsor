# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse

import torch

import funsor
import funsor.torch.distributions as dist
import funsor.ops as ops
from funsor.interpreter import interpretation, reinterpret
from funsor.optimizer import apply_optimizer
from funsor.terms import lazy


def main(args):
    funsor.set_backend("torch")

    # Declare parameters.
    trans_noise = torch.tensor(0.1, requires_grad=True)
    emit_noise = torch.tensor(0.5, requires_grad=True)
    params = [trans_noise, emit_noise]

    # A Gaussian HMM model.
    def model(data):
        log_prob = funsor.to_funsor(0.)

        x_curr = funsor.Tensor(torch.tensor(0.))
        for t, y in enumerate(data):
            x_prev = x_curr

            # A delayed sample statement.
            x_curr = funsor.Variable('x_{}'.format(t), funsor.Real)
            log_prob += dist.Normal(1 + x_prev / 2., trans_noise, value=x_curr)

            # Optionally marginalize out the previous state.
            if t > 0 and not args.lazy:
                log_prob = log_prob.reduce(ops.logaddexp, x_prev.name)

            # An observe statement.
            log_prob += dist.Normal(0.5 + 3 * x_curr, emit_noise, value=y)

        # Marginalize out all remaining delayed variables.
        log_prob = log_prob.reduce(ops.logaddexp)
        return log_prob

    # Train model parameters.
    torch.manual_seed(0)
    data = torch.randn(args.time_steps)
    optim = torch.optim.Adam(params, lr=args.learning_rate)
    for step in range(args.train_steps):
        optim.zero_grad()
        if args.lazy:
            with interpretation(lazy):
                log_prob = apply_optimizer(model(data))
            log_prob = reinterpret(log_prob)
        else:
            log_prob = model(data)
        assert not log_prob.inputs, 'free variables remain'
        loss = -log_prob.data
        loss.backward()
        optim.step()
        if args.verbose and step % 10 == 0:
            print('step {} loss = {}'.format(step, loss.item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Kalman filter example")
    parser.add_argument("-t", "--time-steps", default=10, type=int)
    parser.add_argument("-n", "--train-steps", default=101, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("--lazy", action='store_true')
    parser.add_argument("--filter", action='store_true')
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
