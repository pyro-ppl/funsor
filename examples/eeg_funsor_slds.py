import argparse
from os.path import exists
from urllib.request import urlopen
import time

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict

import funsor
import funsor.distributions as dist
import funsor.ops as ops
from funsor.pyro.convert import matrix_and_mvn_to_funsor, mvn_to_funsor, funsor_to_cat_and_mvn


def download_data():
    if not exists("eeg.dat"):
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"
        with open("eeg.dat", "wb") as f:
            f.write(urlopen(url).read())


class SLDS(nn.Module):
    def __init__(self, num_components, hidden_dim, obs_dim,
                 fine_transition_noise=False, fine_observation_matrix=False,
                 fine_observation_noise=False, fine_transition_matrix=True,
                 moment_matching_lag=2, eval_moment_matching_lag=3):
        self.num_components = num_components
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.moment_matching_lag = moment_matching_lag
        self.eval_moment_matching_lag = eval_moment_matching_lag
        assert moment_matching_lag > 0
        assert eval_moment_matching_lag > 0
        assert fine_transition_noise or fine_observation_matrix or fine_observation_noise or fine_transition_matrix, \
            "The continuous dynamics need to be coupled to the discrete dynamics in at least one way"
        super(SLDS, self).__init__()
        self.transition_logits = nn.Parameter(0.1 * torch.randn(num_components, num_components))
        if fine_transition_matrix:
            transition_matrix = torch.eye(hidden_dim) + 0.05 * torch.randn(num_components, hidden_dim, hidden_dim)
        else:
            transition_matrix = torch.eye(hidden_dim) + 0.05 * torch.randn(hidden_dim, hidden_dim)
        self.transition_matrix = nn.Parameter(transition_matrix)
        if fine_transition_noise:
            self.log_transition_noise = nn.Parameter(0.1 * torch.randn(num_components, hidden_dim))
        else:
            self.log_transition_noise = nn.Parameter(0.1 * torch.randn(hidden_dim))
        if fine_observation_matrix:
            self.observation_matrix = nn.Parameter(0.3 * torch.randn(num_components, hidden_dim, obs_dim))
        else:
            self.observation_matrix = nn.Parameter(0.3 * torch.randn(hidden_dim, obs_dim))
        if fine_observation_noise:
            self.log_obs_noise = nn.Parameter(0.1 * torch.randn(num_components, obs_dim))
        else:
            self.log_obs_noise = nn.Parameter(0.1 * torch.randn(obs_dim))

        x_init_mvn = torch.distributions.MultivariateNormal(torch.zeros(self.hidden_dim), torch.eye(self.hidden_dim))
        self.x_init_mvn = mvn_to_funsor(x_init_mvn, real_inputs=OrderedDict([('x_0', funsor.reals(self.hidden_dim))]))

    def get_tensors_and_dists(self):
        trans_logits = self.transition_logits - self.transition_logits.logsumexp(dim=-1, keepdim=True)
        trans_probs = funsor.Tensor(trans_logits, OrderedDict([("s", funsor.bint(self.num_components))]))

        trans_mvn = torch.distributions.MultivariateNormal(torch.zeros(self.hidden_dim),
                                                           self.log_transition_noise.exp().diag_embed())
        obs_mvn = torch.distributions.MultivariateNormal(torch.zeros(self.obs_dim),
                                                         self.log_obs_noise.exp().diag_embed())

        x_trans_dist = matrix_and_mvn_to_funsor(self.transition_matrix, trans_mvn, ("s",), "x", "y")
        y_dist = matrix_and_mvn_to_funsor(self.observation_matrix, obs_mvn, ("s",), "x", "y")

        return trans_logits, trans_probs, trans_mvn, obs_mvn, x_trans_dist, y_dist

    @funsor.interpreter.interpretation(funsor.terms.moment_matching)
    def log_prob(self, data):
        trans_logits, trans_probs, trans_mvn, obs_mvn, x_trans_dist, y_dist = self.get_tensors_and_dists()

        log_prob = funsor.Number(0.)

        s_vars = {-1: funsor.Tensor(torch.tensor(0), dtype=self.num_components)}
        x_vars = {-1: None}

        for t, y in enumerate(data):
            s_vars[t] = funsor.Variable('s_{}'.format(t), funsor.bint(self.num_components))
            x_vars[t] = funsor.Variable('x_{}'.format(t), funsor.reals(self.hidden_dim))

            log_prob += dist.Categorical(trans_probs(s=s_vars[t - 1]), value=s_vars[t])

            if t == 0:
                log_prob += self.x_init_mvn(value=x_vars[t])
            else:
                log_prob += x_trans_dist(s=s_vars[t], x=x_vars[t - 1], y=x_vars[t])

            if t > self.moment_matching_lag - 1:
                log_prob = log_prob.reduce(ops.logaddexp, frozenset([s_vars[t - self.moment_matching_lag].name,
                                                                     x_vars[t - self.moment_matching_lag].name]))

            log_prob += y_dist(s=s_vars[t], x=x_vars[t], y=y)

        # mop-up
        T = data.shape[0]
        for t in range(self.moment_matching_lag):
            log_prob = log_prob.reduce(ops.logaddexp, frozenset([s_vars[T - self.moment_matching_lag + t].name,
                                                                 x_vars[T - self.moment_matching_lag + t].name]))

        assert not log_prob.inputs, 'unexpected free variables remain'

        return log_prob.data

    @torch.no_grad()
    @funsor.interpreter.interpretation(funsor.terms.moment_matching)
    def filter_and_predict(self, data):
        trans_logits, trans_probs, trans_mvn, obs_mvn, x_trans_dist, y_dist = self.get_tensors_and_dists()

        log_prob = funsor.Number(0.)

        s_vars = {-1: funsor.Tensor(torch.tensor(0), dtype=self.num_components)}
        x_vars = {-1: None}

        filtering_distributions = []
        test_LLs = []

        for t, y in enumerate(data):
            s_vars[t] = funsor.Variable('s_{}'.format(t), funsor.bint(self.num_components))
            x_vars[t] = funsor.Variable('x_{}'.format(t), funsor.reals(self.hidden_dim))

            log_prob += dist.Categorical(trans_probs(s=s_vars[t - 1]), value=s_vars[t])

            if t == 0:
                log_prob += self.x_init_mvn(value=x_vars[t])
            else:
                log_prob += x_trans_dist(s=s_vars[t], x=x_vars[t - 1], y=x_vars[t])

            if t > self.eval_moment_matching_lag - 1:
                log_prob = log_prob.reduce(ops.logaddexp, frozenset([s_vars[t - self.eval_moment_matching_lag].name,
                                                                     x_vars[t - self.eval_moment_matching_lag].name]))

            # filter and compute test LL
            if t > 0:
                srange = range(min(self.eval_moment_matching_lag - 1, t))
                reduce_vars = ["s_{}".format(t - s - 1) for s in srange] + ["x_{}".format(t - s - 1) for s in srange]
                reduction = log_prob.reduce(ops.logaddexp, frozenset(reduce_vars))
                filtering_distributions.append(funsor_to_cat_and_mvn(reduction, 0, ("s_{}".format(t),)))

                reduction = reduction - reduction.reduce(ops.logaddexp)
                test_LLs.append((y_dist(s=s_vars[t], x=x_vars[t], y=y) + reduction).reduce(ops.logaddexp).data.item())

            log_prob += y_dist(s=s_vars[t], x=x_vars[t], y=y)

        # compute test MSE
        means = torch.stack([fd[1].mean for fd in filtering_distributions])  # T-1 2 xdim
        means = torch.matmul(means.unsqueeze(-2), self.observation_matrix).squeeze(-2)  # T-1 2 ydim

        probs = torch.stack([fd[0].logits for fd in filtering_distributions]).exp()
        probs = probs / probs.sum(-1, keepdim=True)  # T-1 2

        means = (probs.unsqueeze(-1) * means).sum(-2)  # T-1 ydim
        mse = (means - data[1:, :]).pow(2.0).mean(-1)

        return mse, torch.tensor(np.array(test_LLs))


def main(args):
    assert args.device in ['cpu', 'gpu']
    print(args)

    download_data()
    data = np.loadtxt('eeg.dat', delimiter=',', skiprows=19)
    print("[raw data shape] ", data.shape)
    data = data[::10, :]
    print("[data shape after thinning] ", data.shape)
    data = data[0:700, :]
    print("[data shape after subselection] ", data.shape)

    labels = data[:, -1].tolist()
    labels = [int(l) for l in labels]

    data = torch.tensor(data[:, :-1]).float()
    data_mean = data.mean(0)
    data -= data_mean
    data_std = data.std(0)
    data /= data_std

    hidden_dim = args.hidden_dim
    T, obs_dim = data.shape

    N_test = 200
    N_train = T - N_test

    print("Length of time series T: {}   Observation dimension: {}".format(T, obs_dim))
    print("N_train: {}  N_test: {}".format(N_train, N_test))

    slds = SLDS(num_components=args.num_components, hidden_dim=hidden_dim, obs_dim=obs_dim,
                fine_observation_noise=args.fon, fine_transition_noise=args.ftn,
                fine_observation_matrix=args.fom, moment_matching_lag=args.moment_matching_lag,
                eval_moment_matching_lag=args.eval_moment_matching_lag)

    if args.load:
        if exists('slds.torch'):
            print('Loading model from slds.torch...')
            slds.load_state_dict(torch.load('slds.torch'))

    if args.device == 'gpu':
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        data = data.cuda()
        slds.cuda()

    adam = torch.optim.Adam(slds.parameters(), lr=args.learning_rate, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(adam, milestones=[50, 150], gamma=0.2)
    ts = [time.time()]

    report_frequency = 10

    for step in range(args.num_steps):
        nll = -slds.log_prob(data[0:N_train, :]) / N_train
        nll.backward()

        adam.step()
        scheduler.step()
        adam.zero_grad()

        ts.append(time.time())
        step_dt = ts[-1] - ts[-2]

        if step % report_frequency == 0 or step == args.num_steps - 1:
            predicted_mse, LLs = slds.filter_and_predict(data[0:N_train + N_test, :])
            predicted_mse = predicted_mse[-N_test:].mean().item()
            test_ll = LLs[-N_test:].mean().item()
            print("[step %03d]  training nll: %.4f   test mse: %.4f  test LL: %.4f \t\t (step_dt: %.2f)" % (step,
                  nll.item(), predicted_mse, test_ll, step_dt))

        if step % 20 == 0 and args.verbose:
            print("[transition logits] mean: %.2f std: %.2f" % (slds.transition_logits.mean().item(),
                                                                slds.transition_logits.std().item()))
            print("[transition logits]\n", slds.transition_logits.data.numpy())
            print("[transition matrix.abs] mean: %.2f std: %.2f" % (slds.transition_matrix.abs().mean().item(),
                                                                    slds.transition_matrix.abs().std().item()))
            print("[transition matrix]\n", slds.transition_matrix.data.numpy())
            print("[log_transition_noise] mean: %.2f std: %.2f" % (slds.log_transition_noise.mean().item(),
                                                                   slds.log_transition_noise.std().item()))
            print("[observation matrix.abs] mean: %.2f std: %.2f" % (slds.observation_matrix.abs().mean().item(),
                                                                     slds.observation_matrix.abs().std().item()))
            print("[log_obs_noise] mean: %.2f std: %.2f  min: %.2f  max: %.2f" % (slds.log_obs_noise.mean().item(),
                                                                                  slds.log_obs_noise.std().item(),
                                                                                  slds.log_obs_noise.min().item(),
                                                                                  slds.log_obs_noise.max().item()))

    if args.save:
        torch.save(slds.state_dict(), 'slds.torch')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Switching linear dynamical system")
    parser.add_argument("-n", "--num-steps", default=2000, type=int)
    parser.add_argument("-hd", "--hidden-dim", default=4, type=int)
    parser.add_argument("-k", "--num-components", default=2, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.15, type=float)
    parser.add_argument("-mml", "--moment-matching-lag", default=1, type=int)
    parser.add_argument("-emml", "--eval-moment-matching-lag", default=2, type=int)
    parser.add_argument("-d", "--device", default="cpu", type=str)
    parser.add_argument("-v", "--verbose", action='store_true')
    parser.add_argument("--fon", action='store_true')
    parser.add_argument("--fom", action='store_true')
    parser.add_argument("--ftn", action='store_true')
    parser.add_argument("--load", action='store_true')
    parser.add_argument("--save", action='store_true')
    args = parser.parse_args()
    main(args)
