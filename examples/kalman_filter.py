from __future__ import absolute_import, division, print_function

import argparse

import torch

import funsor
import funsor.distributions as dist
from funsor.engine.contract_engine import eval as funsor_eval


def main(args):
    # Declare parameters.
    trans_noise = torch.tensor(0.1, requires_grad=True)
    emit_noise = torch.tensor(0.5, requires_grad=True)
    params = [trans_noise, emit_noise]

    # A Gaussian HMM model.
    def model(data):
        log_prob = 0.

        x_curr = 0.
        for t, y in enumerate(data):
            x_prev = x_curr

            # A delayed sample statement.
            x_curr = funsor.Variable('x_{}'.format(t), 'real')
            log_prob += dist.Normal(loc=x_prev, scale=trans_noise).log_prob(x_curr)

            # If we want, we can immediately marginalize out previous sample sites.
            log_prob = funsor_eval(log_prob.logsumexp('x_{}'.format(t - 1)))

            # An observe statement.
            log_prob += dist.Normal(loc=x_curr, scale=emit_noise).log_prob(y)

        return log_prob

    # Train model parameters.
    print('---- training ----')
    data = torch.randn(args.time_steps)
    optim = torch.optim.Adam(params, lr=args.learning_rate)
    for step in range(args.train_steps):
        optim.zero_grad()
        log_prob = model(data)
        loss = -funsor_eval(log_prob.logsumexp())  # Integrates out delayed variables.
        loss.backward()
        optim.step()

    # Serve by drawing a posterior sample.
    print('---- serving ----')
    log_prob = model(data)
    with funsor.adjoints() as result:
        log_prob = funsor_eval(log_prob.logsumexp())  # Forward filter.
    samples = result.backward(log_prob)               # Bakward sample.
    print(samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Kalman filter example")
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
