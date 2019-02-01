from __future__ import absolute_import, division, print_function

import argparse

import torch
import torch.distributions as dist

import funsor


def ConditionalNormal(noise, variance=1000.):
    """
    Approximation to a conditional normal distribution.
    """
    covariance = (torch.eye(2) * variance).requires_grad_()
    covariance[1, 1] += noise
    return dist.MultivariateNormal(torch.zeros(2), covariance)


def main(args):
    # Declare parameters.
    trans_noise = torch.tensor(0.1, requires_grad=True)
    emit_noise = torch.tensor(0.5, requires_grad=True)
    params = [trans_noise, emit_noise]

    def gaussian_hmm(data):
        trans = funsor.MultivariateNormal(
            ('old', 'new'), ('real', 'real'),
            ConditionalNormal(trans_noise))
        emit = funsor.MultivariateNormal(
            ('hidden', 'observed'), ('real', 'real'),
            ConditionalNormal(emit_noise))

        log_prob = 0.
        x_curr = 0.
        for t, y in enumerate(data):
            x_prev, x_curr = x_curr, funsor.var('x_{}'.format(t), 'real')
            log_prob += trans(x_prev, x_curr)
            log_prob += emit(x_curr, y)

        return log_prob.logsumexp()  # performs contraction

    data = torch.randn(args.time_steps)

    optim = torch.optim.Adam(params, lr=args.learning_rate)
    for step in range(args.train_steps):
        optim.zero_grad()
        loss = -gaussian_hmm(data)
        if step % 10 == 0:
            print('step {} loss = {}'.format(step, loss.item()))
        loss.backward()
        optim.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gaussian HMM example")
    parser.add_argument("-t", "--time-steps", default=10, type=int)
    parser.add_argument("-n", "--train-steps", default=101, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("--xfail-if-not-implemented", action='store_true')
    args = parser.parse_args()

    if args.xfail_if_not_implemented:
        try:
            main(args)
        except NotImplementedError:
            print('XFAIL example.py')
    else:
        main(args)
