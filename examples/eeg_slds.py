import argparse
import os
from urllib.request import urlopen
import time

import numpy as np
import torch
import torch.distributions as tdist

from funsor.pyro import SwitchingLinearHMM


def download_data():
    if not os.path.exists("eeg.dat"):
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"
        with open("eeg.dat", "wb") as f:
            f.write(urlopen(url).read())


def clip(params, clip=10.0):
    for p in params:
        p.grad.data.clamp_(min=-clip, max=clip)


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

    K = args.num_components
    hidden_dim = args.hidden_dim
    T, obs_dim = data.shape

    N_test = 100
    N_train = T - 100

    print("Length of time series T: {}   Observation dimension: {}".format(T, obs_dim))
    print("N_train: {}  N_test: {}".format(N_train, N_test))

    if args.device == 'gpu':
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        data = data.cuda()

    initial_logits = 0.1 * torch.randn(K)
    transition_logits = 0.1 * torch.randn(K, K)
    transition_matrix = 0.3 * torch.randn(hidden_dim, hidden_dim)
    log_transition_noise = 0.1 * torch.randn(hidden_dim)
    observation_matrix = 0.3 * torch.randn(hidden_dim, obs_dim)
    log_obs_noise = 0.1 * torch.randn(obs_dim)

    initial_mvn = tdist.MultivariateNormal(torch.zeros(hidden_dim).type_as(initial_logits),
                                           torch.eye(hidden_dim).type_as(initial_logits))

    params = [transition_logits, transition_matrix, log_transition_noise,
              observation_matrix, log_obs_noise]

    for p in params:
        p.requires_grad_(True)

    adam = torch.optim.Adam(params, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(adam, milestones=[25, 50], gamma=0.2)
    ts = [time.time()]

    for step in range(args.num_steps):
        transition_mvn = tdist.MultivariateNormal(torch.zeros(hidden_dim).type_as(initial_logits),
                                                  torch.diag(log_transition_noise.exp()))
        observation_mvn = tdist.MultivariateNormal(torch.zeros(obs_dim).type_as(initial_logits),
                                                   torch.diag(log_obs_noise.exp()))
        slds_dist = SwitchingLinearHMM(initial_logits=initial_logits, initial_mvn=initial_mvn,
                                       transition_logits=transition_logits, transition_matrix=transition_matrix,
                                       transition_mvn=transition_mvn, observation_matrix=observation_matrix,
                                       observation_mvn=observation_mvn)

        nll = -slds_dist.log_prob(data[0:N_train, :]) / N_train
        nll.backward()
        clip(params, clip=args.clip)

        adam.step(), scheduler.step()
        adam.zero_grad()

        ts.append(time.time())
        step_dt = ts[-1] - ts[-2]

        if step % 2 == 0:
            with torch.no_grad():
                ten_step_ll = (slds_dist.log_prob(data[0:N_train + 10, :]) + N_train * nll).item() / 10.0
                hun_step_ll = (slds_dist.log_prob(data[0:N_train + 100, :]) + N_train * nll).item() / 100.0
            print("[step %03d]  training nll: %.4f   test lls: %.4f  %.4f \t\t (step_dt: %.2f)" % (step,
                  nll.item(), ten_step_ll, hun_step_ll, step_dt))

        if step % 20 == 0 and args.verbose:
            print("[transition logits] mean: %.2f std: %.2f" % (transition_logits.mean().item(),
                                                                transition_logits.std().item()))
            print("[transition matrix.abs] mean: %.2f std: %.2f" % (transition_matrix.abs().mean().item(),
                                                                    transition_matrix.abs().std().item()))
            print("[log_transition_noise] mean: %.2f std: %.2f" % (log_transition_noise.mean().item(),
                                                                   log_transition_noise.std().item()))
            print("[observation matrix.abs] mean: %.2f std: %.2f" % (observation_matrix.abs().mean().item(),
                                                                     observation_matrix.abs().std().item()))
            print("[log_obs_noise] mean: %.2f std: %.2f" % (log_obs_noise.mean().item(), log_obs_noise.std().item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Switching linear dynamical system")
    parser.add_argument("-n", "--num-steps", default=100, type=int)
    parser.add_argument("-hd", "--hidden-dim", default=5, type=int)
    parser.add_argument("-k", "--num-components", default=2, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.1, type=float)
    parser.add_argument("-c", "--clip", default=1.0, type=float)
    parser.add_argument("-d", "--device", default="cpu", type=str)
    parser.add_argument("-v", "--verbose", action='store_true')
    args = parser.parse_args()
    main(args)
