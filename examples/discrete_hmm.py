from __future__ import absolute_import, division, print_function

import argparse
from collections import OrderedDict

import torch

import funsor
import funsor.distributions as dist


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

            if isinstance(x_prev, funsor.Variable):
                log_prob = log_prob.logsumexp(x_prev.name)

            log_prob += emit(latent=x_curr, value=y)

        log_prob = log_prob.logsumexp()
        return log_prob

    # Train model parameters.
    print('---- training ----')
    data = torch.ones(args.time_steps)
    optim = torch.optim.Adam(params, lr=args.learning_rate)
    for step in range(args.train_steps):
        optim.zero_grad()
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
