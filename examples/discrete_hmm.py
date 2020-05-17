# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
from collections import OrderedDict

import torch

import funsor
import funsor.torch.distributions as dist
import funsor.ops as ops
from funsor.interpreter import interpretation, reinterpret
from funsor.optimizer import apply_optimizer
from funsor.terms import lazy


def main(args):
    # Declare parameters.
    trans_probs = torch.tensor([[0.2, 0.8], [0.7, 0.3]], requires_grad=True)
    emit_probs = torch.tensor([[0.4, 0.6], [0.1, 0.9]], requires_grad=True)
    params = [trans_probs, emit_probs]

    # A discrete HMM model.
    def model(data):
        log_prob = funsor.to_funsor(0.)

        trans = dist.Categorical(probs=funsor.Tensor(
            trans_probs,
            inputs=OrderedDict([('prev', funsor.bint(args.hidden_dim))]),
        ))

        emit = dist.Categorical(probs=funsor.Tensor(
            emit_probs,
            inputs=OrderedDict([('latent', funsor.bint(args.hidden_dim))]),
        ))

        x_curr = funsor.Number(0, args.hidden_dim)
        for t, y in enumerate(data):
            x_prev = x_curr

            # A delayed sample statement.
            x_curr = funsor.Variable('x_{}'.format(t), funsor.bint(args.hidden_dim))
            log_prob += trans(prev=x_prev, value=x_curr)

            if not args.lazy and isinstance(x_prev, funsor.Variable):
                log_prob = log_prob.reduce(ops.logaddexp, x_prev.name)

            log_prob += emit(latent=x_curr, value=funsor.Tensor(y, dtype=2))

        log_prob = log_prob.reduce(ops.logaddexp)
        return log_prob

    # Train model parameters.
    data = torch.ones(args.time_steps, dtype=torch.long)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Kalman filter example")
    parser.add_argument("-t", "--time-steps", default=10, type=int)
    parser.add_argument("-n", "--train-steps", default=101, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("-d", "--hidden-dim", default=2, type=int)
    parser.add_argument("--lazy", action='store_true')
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
