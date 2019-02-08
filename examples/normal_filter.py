from __future__ import absolute_import, division, print_function

import argparse

import torch

import funsor
import funsor.distributions as dist


def main(args_):
    # declare parameters
    trans_noise = torch.tensor(0.1, requires_grad=True)
    emit_noise = torch.tensor(0.5, requires_grad=True)
    params = [trans_noise, emit_noise]

    # this is like pyro.sample, but uses args_ rather than poutine
    def pyro_sample(name, fn, obs=None):
        if obs is not None:
            value = obs
        elif args_.eager:
            # this is eager like usual pyro.sample
            value = fn.sample(['value'])
        else:
            # this is deferred like pyro.sample during enumeration
            value = funsor.Variable(name, fn.schema['value'])
        log_prob = fn(value=value)
        return value, log_prob

    # a Gaussian HMM model
    def model(data):
        args = {}
        log_prob_sum = 0.
        x_curr = 0.
        for t, y in enumerate(data):
            x_prev = x_curr

            # a sample statement
            x_curr, log_prob = pyro_sample(
                'x_{}'.format(t),
                dist.Normal(loc=x_prev, scale=trans_noise))
            log_prob_sum += log_prob

            # an observe statement
            _, log_prob = pyro_sample(
                'y_{}'.format(t),
                dist.Normal(loc=x_curr, scale=emit_noise),
                obs=y)
            log_prob_sum += log_prob

            # note filtering only makes sense in deferred mode.
            # this could be made safe via something like:
            #   x_curr = pyro.barrier(x_curr)
            if args_.filter:
                # perform a filter update
                args_t = log_prob_sum.argmax()
                log_prob_sum = log_prob_sum.max()
                args.update(args_t)
                x_curr = args[x_curr.name]

        return args, log_prob_sum

    data = torch.randn(args_.time_steps)

    # training loop
    print('---- training ----')
    optim = torch.optim.Adam(params, lr=args_.learning_rate)
    for step in range(args_.train_steps):
        optim.zero_grad()
        eager_args, log_prob = model(data)
        loss = -funsor.eval(log_prob.logsumexp())  # integrates out deferred variables
        if step % 10 == 0:
            print('step {} loss = {}'.format(step, loss.item()))
        loss.backward()
        optim.step()

    # serving by drawing a posterior sample
    print('---- serving ----')
    eager_args = model(data)
    lazy_args = funsor.eval(log_prob.sample())
    joint_sample = eager_args
    joint_sample.update(lazy_args)
    for key, value in sorted(joint_sample.items()):
        print('{} = {}'.format(key, value.item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gaussian HMM example")
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
