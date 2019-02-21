from __future__ import absolute_import, division, print_function

import argparse

import torch

import funsor
import funsor.distributions as dist


def main(args):
    # Declare parameters.
    trans_noise = torch.tensor(0.1, requires_grad=True)
    emit_noise = torch.tensor(0.5, requires_grad=True)
    params = [trans_noise, emit_noise]

    # A Gaussian HMM model.
    def model(data):
        prob = 1.

        x_curr = 0.
        for t, y in enumerate(data):
            x_prev = x_curr

            # A delayed sample statement.
            x_curr = funsor.Variable('x_{}'.format(t), funsor.reals())
            prob *= dist.Normal(loc=x_prev, scale=trans_noise, value=x_curr)

            # If we want, we can immediately marginalize out previous sample sites.
            prob = prob.sum('x_{}'.format(t - 1))
            # TODO prob = Clever(funsor.eval)(prob)

            # An observe statement.
            prob *= dist.Normal(loc=x_curr, scale=emit_noise, value=y)

        return prob

    # Train model parameters.
    print('---- training ----')
    data = torch.randn(args.time_steps)
    optim = torch.optim.Adam(params, lr=args.learning_rate)
    for step in range(args.train_steps):
        optim.zero_grad()
        prob = model(data)
        # TODO prob = Clever(funsor.eval)(prob)
        loss = -prob.sum().log()  # Integrates out delayed variables.
        loss.backward()
        optim.step()

    # Serve by drawing a posterior sample.
    print('---- serving ----')
    prob = model(data)
    prob = funsor.eval(prob.sum())         # Forward filter.
    samples = prob.backward(prob.log())    # Bakward sample.
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
