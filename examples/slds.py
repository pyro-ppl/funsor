from __future__ import absolute_import, division, print_function

import argparse

import torch

import funsor
import funsor.distributions as dist
import funsor.ops as ops


def main(args):
    # Declare parameters.
    trans_probs = funsor.Tensor(torch.tensor([[0.9, 0.1],
                                              [0.1, 0.9]], requires_grad=True))
    trans_noise = funsor.Tensor(torch.tensor([
        0.1,  # low noise component
        0.1,  # low noise component
        # 1.0,  # high noisy component
    ], requires_grad=True))
    emit_noise = funsor.Tensor(torch.tensor(0.5, requires_grad=True))
    params = [trans_probs.data,
              trans_noise.data,
              emit_noise.data]

    # A Gaussian HMM model.
    @funsor.interpreter.interpretation(funsor.terms.moment_matching)
    def model(data):
        log_prob = funsor.Number(0.)

        # s is the discrete latent state,
        # x is the continuous latent state,
        # y is the observed state.
        s_curr = funsor.Tensor(torch.tensor(0), dtype=2)
        x_curr = funsor.Tensor(torch.tensor(0.))
        for t, y in enumerate(data):
            s_prev = s_curr
            x_prev = x_curr

            # A delayed sample statement.
            s_curr = funsor.Variable('s_{}'.format(t), funsor.bint(2))
            log_prob += dist.Categorical(trans_probs[s_prev], value=s_curr)

            # A delayed sample statement.
            x_curr = funsor.Variable('x_{}'.format(t), funsor.reals())
            log_prob += dist.Normal(x_prev, trans_noise[s_curr], value=x_curr)

            # Marginalize out previous delayed sample statements.
            if t > 0:
                log_prob = log_prob.reduce(
                    ops.logaddexp, frozenset([s_prev.name, x_prev.name]))

            # An observe statement.
            log_prob += dist.Normal(x_curr, emit_noise, value=y)

        log_prob = log_prob.reduce(ops.logaddexp)
        return log_prob

    # Train model parameters.
    torch.manual_seed(0)
    data = torch.randn(args.time_steps)
    optim = torch.optim.Adam(params, lr=args.learning_rate)
    for step in range(args.train_steps):
        optim.zero_grad()
        log_prob = model(data)
        assert not log_prob.inputs, 'free variables remain'
        loss = -log_prob.data
        loss.backward()
        optim.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Switching linear dynamical system")
    parser.add_argument("-t", "--time-steps", default=10, type=int)
    parser.add_argument("-n", "--train-steps", default=101, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("--filter", action='store_true')
    args = parser.parse_args()
    main(args)
