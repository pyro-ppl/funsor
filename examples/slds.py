import argparse
import time
from collections import OrderedDict

import torch

import funsor
import funsor.distributions as dist
import funsor.ops as ops


def main(args):
    if args.cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    num_comp = args.num_components
    hidden_dim = args.hidden_dim
    obs_dim = args.obs_dim

    # TODO replace these with matrix_and_mvn_to_funsor()
    # as in SwitchingLinearHMM.__init__().
    @funsor.torch.function(funsor.reals(hidden_dim),
                           funsor.reals(hidden_dim, hidden_dim),
                           funsor.reals(hidden_dim))
    def trans_mm(vector, matrix):
        return vector.unsqueeze(-2).matmul(matrix).squeeze(-2)

    @funsor.torch.function(funsor.reals(hidden_dim),
                           funsor.reals(hidden_dim, obs_dim),
                           funsor.reals(obs_dim))
    def obs_mm(vector, matrix):
        return vector.unsqueeze(-2).matmul(matrix).squeeze(-2)

    # Declare parameters.
    s_inputs = OrderedDict([("s", funsor.bint(num_comp))])  # for class-dependent parameters

    trans_probs = funsor.Tensor(torch.eye(num_comp), s_inputs)

    trans_noise = funsor.Tensor(torch.eye(hidden_dim).expand(num_comp, -1, -1), s_inputs)
    trans_matrix = funsor.Tensor(torch.eye(hidden_dim).expand(num_comp, -1, -1), s_inputs)

    obs_matrix = funsor.Tensor(torch.rand(num_comp, hidden_dim, obs_dim), s_inputs)
    obs_noise = funsor.Tensor(torch.eye(obs_dim).expand(num_comp, -1, -1), s_inputs)

    params = [trans_matrix.data, obs_matrix.data]
    for p in params:
        p.requires_grad_()

    # A Gaussian HMM model.
    @funsor.interpreter.interpretation(funsor.terms.moment_matching)
    def model(data):
        log_prob = funsor.Number(0.)

        # s is the discrete latent state,
        # x is the continuous latent state,
        # y is the observed state.
        s_curr = funsor.Tensor(torch.tensor(0), dtype=num_comp)
        x_curr = funsor.Tensor(torch.zeros(hidden_dim))

        for t, y in enumerate(data):
            s_prev = s_curr
            x_prev = x_curr

            # A delayed sample statement.
            s_curr = funsor.Variable('s_{}'.format(t), funsor.bint(num_comp))
            log_prob += dist.Categorical(trans_probs(s=s_prev), value=s_curr)

            # A delayed sample statement.
            x_curr = funsor.Variable('x_{}'.format(t), funsor.reals(hidden_dim))
            trans = trans_matrix(s=s_curr)
            log_prob += dist.MultivariateNormal(
                trans_mm(x_prev, trans),
                trans_noise(s=s_curr), value=x_curr)

            # Marginalize out previous delayed sample statements.
            if t > 0:
                log_prob = log_prob.reduce(
                    ops.logaddexp, frozenset([s_prev.name, x_prev.name]))

            # An observe statement.
            log_prob += dist.MultivariateNormal(
                obs_mm(x_curr, obs_matrix(s=s_curr)),
                obs_noise(s=s_curr), value=y)

        log_prob = log_prob.reduce(ops.logaddexp)
        return log_prob

    # Train model parameters.
    torch.manual_seed(0)
    data = torch.randn(args.time_steps, obs_dim)
    optim = torch.optim.Adam(params, lr=args.learning_rate)

    ts = [time.time()]

    for step in range(args.train_steps):
        optim.zero_grad()
        log_prob = model(data)
        assert not log_prob.inputs, 'free variables remain'
        assert isinstance(log_prob, funsor.Tensor)
        loss = -log_prob.data
        loss.backward()
        optim.step()
        ts.append(time.time())
        if args.verbose and step % 10 == 0:
            dt = ts[-1] - ts[-2]
            print('step {} loss = {:4f}    step_dt: {:3f}'.format(step, loss.item(), dt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Switching linear dynamical system")
    parser.add_argument("-t", "--time-steps", default=100, type=int)
    parser.add_argument("-n", "--train-steps", default=101, type=int)
    parser.add_argument("-hd", "--hidden-dim", default=2, type=int)
    parser.add_argument("-od", "--obs-dim", default=2, type=int)
    parser.add_argument("-k", "--num-components", default=2, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
