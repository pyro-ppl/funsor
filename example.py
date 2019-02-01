from __future__ import absolute_import, division, print_function

import argparse

import torch
import torch.distributions as dist

import funsor


def main(args):
    # declare parameters
    trans_noise = torch.tensor([0.1], requires_grad=True)
    emit_noise = torch.tensor([0.5], requires_grad=True)
    params = [trans_noise, emit_noise]

    # a Gaussian HMM model
    def model(data):
        trans = funsor.Normal(
            ('drift',), ('real',),
            dist.Independent(dist.Normal(0., trans_noise), 1))
        emit = funsor.Normal(
            ('error',), ('real',),
            dist.Independent(dist.Normal(0., emit_noise), 1))

        log_prob = 0.
        x_curr = 0.
        for t, y in enumerate(data):
            x_prev, x_curr = x_curr, funsor.var('x_{}'.format(t), 'real')
            log_prob += trans(x_prev - x_curr)
            log_prob += emit(x_curr - y)

        return log_prob

    data = torch.randn(args.time_steps)

    # training loop
    print('---- training ----')
    optim = torch.optim.Adam(params, lr=args.learning_rate)
    for step in range(args.train_steps):
        optim.zero_grad()
        log_prob = model(data)
        loss = -log_prob.logsumexp()  # performs contraction
        if step % 10 == 0:
            print('step {} loss = {}'.format(step, loss.item()))
        loss.backward()
        optim.step()

    # serving by drawing a posterior sample
    print('---- serving ----')
    log_prob = model(data)
    joint_sample, log_prob = log_prob.sample()
    for key, value in sorted(joint_sample.items()):
        print('{} = {}'.format(key, value.item()))


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
