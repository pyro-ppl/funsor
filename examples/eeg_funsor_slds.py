import argparse
from os.path import exists
from urllib.request import urlopen
import time

import numpy as np
import torch
import torch.distributions as tdist
import torch.nn as nn

from collections import OrderedDict

import funsor
import funsor.distributions as dist
import funsor.ops as ops
from funsor.gaussian import Gaussian
from funsor.pyro.convert import matrix_and_mvn_to_funsor


def download_data():
    if not exists("eeg.dat"):
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"
        with open("eeg.dat", "wb") as f:
            f.write(urlopen(url).read())


def clip(params, clip=10.0):
    for p in params:
        p.grad.data.clamp_(min=-clip, max=clip)


class SLDS(nn.Module):
    def __init__(self, num_components, hidden_dim, obs_dim):
        self.num_components = num_components
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        super(SLDS, self).__init__()
        #self.initial_logits = nn.Parameter(0.1 * torch.randn(num_components))
        self.transition_logits = nn.Parameter(0.1 * torch.randn(num_components, num_components))
        transition_matrix = 0.03 * torch.randn(hidden_dim, hidden_dim) + torch.eye(hidden_dim)
        self.transition_matrix = nn.Parameter(transition_matrix.expand(num_components, -1, -1))
        self.log_transition_noise = nn.Parameter(0.1 * torch.randn(hidden_dim))
        #self.log_transition_noise = nn.Parameter(0.1 * torch.randn(num_components, hidden_dim))
        self.observation_matrix = nn.Parameter(0.3 * torch.randn(hidden_dim, obs_dim))
        #self.observation_matrix = nn.Parameter(0.3 * torch.randn(num_components, hidden_dim, obs_dim))
        self.log_obs_noise = nn.Parameter(0.1 * torch.randn(obs_dim))
        #self.log_obs_noise = nn.Parameter(0.1 * torch.randn(num_components, obs_dim))

    def log_prob(self, data):
        trans_logits = self.transition_logits - self.transition_logits.logsumexp(dim=-1, keepdim=True)
        trans_probs = funsor.Tensor(trans_logits, OrderedDict([("s", funsor.bint(self.num_components))]))

        log_prob = funsor.Number(0.)

        s_curr = funsor.Tensor(torch.tensor(0), dtype=self.num_components)
        x_curr = None
        x_init_mvn = Gaussian(torch.zeros(self.hidden_dim), torch.eye(self.hidden_dim),
                              inputs=OrderedDict([('x_0', funsor.reals(self.hidden_dim))]))
        trans_mvn = torch.distributions.MultivariateNormal(torch.zeros(self.hidden_dim),
                                                           self.log_transition_noise.exp().diag_embed())
        obs_mvn = torch.distributions.MultivariateNormal(torch.zeros(self.obs_dim),
                                                         self.log_obs_noise.exp().diag_embed())

        for t, y in enumerate(data):
            s_prev = s_curr
            x_prev = x_curr

            s_curr = funsor.Variable('s_{}'.format(t), funsor.bint(self.num_components))
            log_prob += dist.Categorical(trans_probs(s=s_prev), value=s_curr)

            x_curr = funsor.Variable('x_{}'.format(t), funsor.reals(self.hidden_dim))
            if t == 0:
                x_dist = x_init_mvn
            else:
                x_dist = matrix_and_mvn_to_funsor(self.transition_matrix, trans_mvn,
                                                  ("s_{}".format(t),),
                                                  x_name="x_{}".format(t - 1), y_name="x_{}".format(t))
            log_prob += x_dist(value=x_curr)

            if t > 0:
                log_prob = log_prob.reduce(ops.logaddexp, frozenset([s_prev.name, x_prev.name]))

            y_dist = matrix_and_mvn_to_funsor(self.observation_matrix, obs_mvn,
                                              ("s_{}".format(t),), x_name="x_{}".format(t), y_name="value")
            log_prob += y_dist(value=y)

        log_prob = log_prob.reduce(ops.logaddexp)
        assert not log_prob.inputs, 'free variables remain'

        return log_prob.data

    def filter(self, value):
        raise NotImplementedError


def main(args):
    assert args.device in ['cpu', 'gpu']
    print(args)

    download_data()
    data = np.loadtxt('eeg.dat', delimiter=',', skiprows=19)
    # labels = data[:, -1]

    data = torch.tensor(data[:, :-1]).float()
    data_mean = data.mean(0)
    data -= data_mean
    data_std = data.std(0)
    data /= data_std

    data = data[0:50, :]

    hidden_dim = args.hidden_dim
    T, obs_dim = data.shape

    N_test = 0
    N_train = T - N_test

    print("Length of time series T: {}   Observation dimension: {}".format(T, obs_dim))
    print("N_train: {}  N_test: {}".format(N_train, N_test))

    slds = SLDS(num_components=args.num_components, hidden_dim=hidden_dim, obs_dim=obs_dim)

    if 0:
        if exists('slds.torch'):
            print('Loading model from slds.torch...')
            slds.load_state_dict(torch.load('slds.torch'))

    if args.device == 'gpu':
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        data = data.cuda()
        slds.cuda()

    adam = torch.optim.Adam(slds.parameters(), lr=0.3, amsgrad=True)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(adam, milestones=[20, 40, 80], gamma=0.2)
    opt = torch.optim.LBFGS(slds.parameters(), lr=args.learning_rate)  # line_search_fn='strong_wolfe')
    ts = [time.time()]

    for step in range(args.num_steps):
        if step > 90:
            def closure():
                opt.zero_grad()
                loss = -slds.log_prob(data[0:N_train, :]) / N_train
                loss.backward(retain_graph=True)
                return loss
            nll = opt.step(closure)
        else:
            nll = -slds.log_prob(data[0:N_train, :]) / N_train
            nll.backward()

        # clip(params, clip=args.clip)
        # opt.zero_grad()

            adam.step()  # scheduler.step()
            adam.zero_grad()

        ts.append(time.time())
        step_dt = ts[-1] - ts[-2]

        if step % 1 == 0 or step == args.num_steps - 1:
            print("[step %03d]  training nll: %.4f   test lls: %.4f  %.4f \t\t (step_dt: %.2f)" % (step,
                  nll.item(), 0.0, 0.0, step_dt))

        if step % 20 == 0 and args.verbose:
            print("[transition logits] mean: %.2f std: %.2f" % (slds.transition_logits.mean().item(),
                                                                slds.transition_logits.std().item()))
            print("[transition matrix.abs] mean: %.2f std: %.2f" % (slds.transition_matrix.abs().mean().item(),
                                                                    slds.transition_matrix.abs().std().item()))
            print("[log_transition_noise] mean: %.2f std: %.2f" % (slds.log_transition_noise.mean().item(),
                                                                   slds.log_transition_noise.std().item()))
            print("[observation matrix.abs] mean: %.2f std: %.2f" % (slds.observation_matrix.abs().mean().item(),
                                                                     slds.observation_matrix.abs().std().item()))
            print("[log_obs_noise] mean: %.2f std: %.2f" % (slds.log_obs_noise.mean().item(),
                                                            slds.log_obs_noise.std().item()))

    # torch.save(slds.state_dict(), 'slds.torch')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Switching linear dynamical system")
    parser.add_argument("-n", "--num-steps", default=1000, type=int)
    parser.add_argument("-hd", "--hidden-dim", default=4, type=int)
    parser.add_argument("-k", "--num-components", default=2, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.1, type=float)
    parser.add_argument("-c", "--clip", default=1.0, type=float)
    parser.add_argument("-d", "--device", default="cpu", type=str)
    parser.add_argument("-v", "--verbose", action='store_true')
    args = parser.parse_args()
    main(args)
