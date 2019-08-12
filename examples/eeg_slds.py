import argparse
import os
from os.path import exists
from urllib.request import urlopen
import time

import numpy as np
import torch
import torch.distributions as tdist
import torch.nn as nn

from funsor.pyro import SwitchingLinearHMM


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
        self.initial_logits = nn.Parameter(0.1 * torch.randn(num_components))
        self.transition_logits = nn.Parameter(0.1 * torch.randn(num_components, num_components))
        self.transition_matrix = nn.Parameter(0.03 * torch.randn(hidden_dim, hidden_dim) + torch.eye(hidden_dim))
        self.log_transition_noise = nn.Parameter(0.1 * torch.randn(hidden_dim))
        self.observation_matrix = nn.Parameter(0.3 * torch.randn(hidden_dim, obs_dim))
        self.log_obs_noise = nn.Parameter(0.1 * torch.randn(obs_dim))
        self.initial_mvn = None

    def log_prob(self, value):
        if self.initial_mvn is None:
            initial_mvn_mean = torch.zeros(self.hidden_dim).type_as(value)
            initial_mvn_cov = torch.eye(self.hidden_dim).type_as(value)
            self.initial_mvn = tdist.MultivariateNormal(initial_mvn_mean, initial_mvn_cov)

        transition_mvn = tdist.MultivariateNormal(torch.zeros(self.hidden_dim).type_as(value),
                                                  torch.diag(self.log_transition_noise.exp()))
        observation_mvn = tdist.MultivariateNormal(torch.zeros(self.obs_dim).type_as(value),
                                                   torch.diag(self.log_obs_noise.exp()))
        slds_dist = SwitchingLinearHMM(initial_logits=self.initial_logits,
                                       initial_mvn=self.initial_mvn,
                                       transition_logits=self.transition_logits,
                                       transition_matrix=self.transition_matrix,
                                       transition_mvn=transition_mvn,
                                       observation_matrix=self.observation_matrix,
                                       observation_mvn=observation_mvn)

        return slds_dist.log_prob(value)


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

    data = data[0:600, :]

    hidden_dim = args.hidden_dim
    T, obs_dim = data.shape

    N_test = 100
    N_train = T - N_test

    assert N_train % args.num_splits == 0

    print("Length of time series T: {}   Observation dimension: {}".format(T, obs_dim))
    print("N_train: {}  N_test: {}".format(N_train, N_test))

    slds = SLDS(num_components=args.num_components, hidden_dim=hidden_dim, obs_dim=obs_dim)
    if exists('slds.torch'):
        print('Loading model from slds.torch...')
        slds.load_state_dict(torch.load('slds.torch'))

    if args.device == 'gpu':
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        data = data.cuda()
        slds.cuda()

    adam = torch.optim.Adam(slds.parameters(), lr=0.3, amsgrad=True)
    #adam = torch.optim.Adam(params, lr=args.learning_rate, amsgrad=True)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(adam, milestones=[20, 40, 80], gamma=0.2)
    opt = torch.optim.LBFGS(slds.parameters(), lr=args.learning_rate) #, line_search_fn='strong_wolfe')
    ts = [time.time()]

    for step in range(args.num_steps):
        if step > 30:
            def closure():
                opt.zero_grad()
                loss = -slds.log_prob(data[0:N_train, :]) / N_train
                loss.backward(retain_graph=True)
                return loss
            nll = opt.step(closure)
        else:
            nll = -slds.log_prob(data[0:N_train, :]) / N_train
            nll.backward()
        #clip(params, clip=args.clip)
        #opt.zero_grad()

            adam.step() #, scheduler.step()
            adam.zero_grad()

        ts.append(time.time())
        step_dt = ts[-1] - ts[-2]

        if step % 5 == 0:
            with torch.no_grad():
                zer_step_ll = slds.log_prob(data[0:N_train, :]).item()
                ten_step_ll = (slds.log_prob(data[0:N_train + 10, :]) - zer_step_ll).item() / 10.0
                hun_step_ll = (slds.log_prob(data[0:N_train + 180, :]) - zer_step_ll).item() / 180.0
            print("[step %03d]  training nll: %.4f   test lls: %.4f  %.4f \t\t (step_dt: %.2f)" % (step,
                  nll.item(), ten_step_ll, hun_step_ll, step_dt))

        if step % 20 == 0 and args.verbose:
            print("[transition logits] mean: %.2f std: %.2f" % (slds.transition_logits.mean().item(),
                                                                slds.transition_logits.std().item()))
            print("[transition matrix.abs] mean: %.2f std: %.2f" % (slds.transition_matrix.abs().mean().item(),
                                                                    slds.transition_matrix.abs().std().item()))
            print("[log_transition_noise] mean: %.2f std: %.2f" % (slds.log_transition_noise.mean().item(),
                                                                   slds.log_transition_noise.std().item()))
            print("[observation matrix.abs] mean: %.2f std: %.2f" % (slds.observation_matrix.abs().mean().item(),
                                                                     slds.observation_matrix.abs().std().item()))
            print("[log_obs_noise] mean: %.2f std: %.2f" % (slds.log_obs_noise.mean().item(), slds.log_obs_noise.std().item()))

    torch.save(slds.state_dict(), 'slds.torch')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Switching linear dynamical system")
    parser.add_argument("-n", "--num-steps", default=1000, type=int)
    parser.add_argument("-s", "--num-splits", default=1, type=int)
    parser.add_argument("-hd", "--hidden-dim", default=5, type=int)
    parser.add_argument("-k", "--num-components", default=2, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.1, type=float)
    parser.add_argument("-c", "--clip", default=1.0, type=float)
    parser.add_argument("-d", "--device", default="gpu", type=str)
    parser.add_argument("-v", "--verbose", action='store_true')
    args = parser.parse_args()
    main(args)
