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


def clip(params, clip=10.0):
    for p in params:
        p.grad.data.clamp_(min=-clip, max=clip)


class SLDS(nn.Module):
    def __init__(self, num_components, hidden_dim, obs_dim,
                 fine_transition_noise=False, fine_observation_matrix=False,
                 fine_observation_noise=False,
                 moment_matching_lag=2):
        self.num_components = num_components
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.moment_matching_lag = moment_matching_lag
        assert moment_matching_lag > 0
        super(SLDS, self).__init__()
        self.transition_logits = nn.Parameter(0.1 * torch.randn(num_components, num_components))
        transition_matrix = torch.eye(hidden_dim) + 0.05 * torch.randn(num_components, hidden_dim, hidden_dim)
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

    @funsor.interpreter.interpretation(funsor.terms.moment_matching)
    def log_prob(self, data, filtering=False):
        trans_logits = self.transition_logits - self.transition_logits.logsumexp(dim=-1, keepdim=True)
        trans_probs = funsor.Tensor(trans_logits, OrderedDict([("s", funsor.bint(self.num_components))]))

        trans_mvn = torch.distributions.MultivariateNormal(torch.zeros(self.hidden_dim),
                                                           self.log_transition_noise.exp().diag_embed())
        obs_mvn = torch.distributions.MultivariateNormal(torch.zeros(self.obs_dim),
                                                         self.log_obs_noise.exp().diag_embed())

        x_trans_dist = matrix_and_mvn_to_funsor(self.transition_matrix, trans_mvn, ("s",), "x", "y")
        y_dist = matrix_and_mvn_to_funsor(self.observation_matrix, obs_mvn, ("s",), "x", "y")

        log_prob = funsor.Number(0.)

        s_vars = {0: funsor.Tensor(torch.tensor(0), dtype=self.num_components)}
        x_vars = {0: None}
        filtering_distributions = []

        for t, y in enumerate(data):
            s_curr = funsor.Variable('s_{}'.format(t), funsor.bint(self.num_components))
            x_curr = funsor.Variable('x_{}'.format(t), funsor.reals(self.hidden_dim))
            s_vars[t + 1] = s_curr
            x_vars[t + 1] = x_curr

            log_prob += dist.Categorical(trans_probs(s=s_vars[t]), value=s_curr)

            if t == 0:
                log_prob += self.x_init_mvn(value=x_curr)
            else:
                log_prob += x_trans_dist(s=s_curr, x=x_vars[t], y=x_curr)

            if t > self.moment_matching_lag - 1:
                log_prob = log_prob.reduce(ops.logaddexp, frozenset([s_vars[t - self.moment_matching_lag + 1].name,
                                                                     x_vars[t - self.moment_matching_lag + 1].name]))
            if filtering and t > 0:
                reduce_vars = ["s_{}".format(t - s - 1) for s in range(self.moment_matching_lag - 1)]
                reduce_vars += ["x_{}".format(t - s - 1) for s in range(self.moment_matching_lag - 1)]
                reduce_vars = frozenset(reduce_vars)
                filtering_distributions.append(funsor_to_cat_and_mvn(log_prob.reduce(ops.logaddexp, reduce_vars),
                                                                     0, ("s_{}".format(t),)))

            log_prob += y_dist(s=s_curr, x=x_curr, y=y)

        # mop-up
        T = data.shape[0]
        for t in range(self.moment_matching_lag):
            log_prob = log_prob.reduce(ops.logaddexp, frozenset([s_vars[T - self.moment_matching_lag + t + 1].name,
                                                                 x_vars[T - self.moment_matching_lag + t + 1].name]))

        assert not log_prob.inputs, 'free variables remain'

        return log_prob.data, filtering_distributions


def main(args):
    assert args.device in ['cpu', 'gpu']
    print(args)

    download_data()
    data = np.loadtxt('eeg.dat', delimiter=',', skiprows=19)
    data = data[1000:1800, :]
    labels = data[:, -1].tolist()
    labels = [int(l) for l in labels]

    first_one = None
    last_one = None
    for k, x in enumerate(labels):
        if x == 1 and first_one is None:
            first_one = k
        if last_one is None and first_one is not None and x == 0:
            last_one = k

    data = torch.tensor(data[:, :-1]).float()
    data_mean = data.mean(0)
    data -= data_mean
    data_std = data.std(0)
    data /= data_std

    hidden_dim = args.hidden_dim
    T, obs_dim = data.shape

    N_test = 0
    N_train = T - N_test

    print("Length of time series T: {}   Observation dimension: {}".format(T, obs_dim))
    print("N_train: {}  N_test: {}".format(N_train, N_test))

    slds = SLDS(num_components=args.num_components, hidden_dim=hidden_dim, obs_dim=obs_dim,
                fine_observation_noise=args.fon, fine_transition_noise=args.ftn,
                fine_observation_matrix=args.fom, moment_matching_lag=args.moment_matching_lag)

    if 0:
        if exists('slds.torch'):
            print('Loading model from slds.torch...')
            slds.load_state_dict(torch.load('slds.torch'))

    if args.device == 'gpu':
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        data = data.cuda()
        slds.cuda()

    adam = torch.optim.Adam(slds.parameters(), lr=args.learning_rate, amsgrad=True)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(adam, milestones=[20, 40, 80], gamma=0.2)
    opt = torch.optim.LBFGS(slds.parameters(), lr=args.learning_rate)  # line_search_fn='strong_wolfe')
    ts = [time.time()]

    for step in range(args.num_steps):
        if step > 9999:
            def closure():
                opt.zero_grad()
                loss = -slds.log_prob(data[0:N_train, :]) / N_train
                loss.backward(retain_graph=True)
                return loss
            nll = opt.step(closure)
        else:
            filtering = True if step % 50 == 0 else False
            nll, filtering_dists = slds.log_prob(data[0:N_train, :], filtering=filtering)
            nll = -nll / N_train
            nll.backward()
            if filtering:
                filtering_dists = [np.argmax(fl[0].logits.data.numpy()) for fl in filtering_dists]
                print("labels[0:first_one]", labels[0:first_one])
                print("filtering_dists[0:first_one]", filtering_dists[0:first_one])
                print("labels[first_one:last_one]", labels[first_one:last_one])
                print("filtering_dists[first_one:last_one]", filtering_dists[first_one:last_one])
                print("labels[last_one:]", labels[last_one:])
                print("filtering_dists[last_one:]", filtering_dists[last_one:])

        # clip(params, clip=args.clip)
        # opt.zero_grad()

            adam.step()  # scheduler.step()
            adam.zero_grad()

        ts.append(time.time())
        step_dt = ts[-1] - ts[-2]

        if step % 2 == 0 or step == args.num_steps - 1:
            print("[step %03d]  training nll: %.4f   test lls: %.4f  %.4f \t\t (step_dt: %.2f)" % (step,
                  nll.item(), 0.0, 0.0, step_dt))

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

    # torch.save(slds.state_dict(), 'slds.torch')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Switching linear dynamical system")
    parser.add_argument("-n", "--num-steps", default=2000, type=int)
    parser.add_argument("-hd", "--hidden-dim", default=8, type=int)
    parser.add_argument("-k", "--num-components", default=2, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.20, type=float)
    parser.add_argument("-mml", "--moment-matching-lag", default=1, type=int)
    parser.add_argument("-c", "--clip", default=1.0, type=float)
    parser.add_argument("-d", "--device", default="cpu", type=str)
    parser.add_argument("-v", "--verbose", action='store_true')
    parser.add_argument("--fon", action='store_true')
    parser.add_argument("--fom", action='store_true')
    parser.add_argument("--ftn", action='store_true')
    args = parser.parse_args()
    main(args)
