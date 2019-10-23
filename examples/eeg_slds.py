"""
We use a switching linear dynamical system [1] to model a EEG time series dataset.
For inference we use a moment-matching approximation enabled by
`funsor.interpreter.interpretation(funsor.terms.moment_matching)`.

References

[1] Anderson, B., and J. Moore. "Optimal filtering. Prentice-Hall, Englewood Cliffs." New Jersey (1979).
"""
import argparse
import time
from collections import OrderedDict
from os.path import exists
from urllib.request import urlopen

import numpy as np
import torch
import torch.nn as nn

import funsor
import funsor.distributions as dist
import funsor.ops as ops
from funsor.pyro.convert import funsor_to_cat_and_mvn, funsor_to_mvn, matrix_and_mvn_to_funsor, mvn_to_funsor


# download dataset from UCI archive
def download_data():
    if not exists("eeg.dat"):
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"
        with open("eeg.dat", "wb") as f:
            f.write(urlopen(url).read())


class SLDS(nn.Module):
    def __init__(self,
                 num_components,   # the number of switching states K
                 hidden_dim,       # the dimension of the continuous latent space
                 obs_dim,          # the dimension of the continuous outputs
                 fine_transition_matrix=True,    # controls whether the transition matrix depends on s_t
                 fine_transition_noise=False,    # controls whether the transition noise depends on s_t
                 fine_observation_matrix=False,  # controls whether the observation matrix depends on s_t
                 fine_observation_noise=False,   # controls whether the observation noise depends on s_t
                 moment_matching_lag=1):         # controls the expense of the moment matching approximation

        self.num_components = num_components
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.moment_matching_lag = moment_matching_lag
        self.fine_transition_noise = fine_transition_noise
        self.fine_observation_matrix = fine_observation_matrix
        self.fine_observation_noise = fine_observation_noise
        self.fine_transition_matrix = fine_transition_matrix

        assert moment_matching_lag > 0
        assert fine_transition_noise or fine_observation_matrix or fine_observation_noise or fine_transition_matrix, \
            "The continuous dynamics need to be coupled to the discrete dynamics in at least one way [use at " + \
            "least one of the arguments --ftn --ftm --fon --fom]"

        super(SLDS, self).__init__()

        # initialize the various parameters of the model
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

        # define the prior distribution p(x_0) over the continuous latent at the initial time step t=0
        x_init_mvn = torch.distributions.MultivariateNormal(torch.zeros(self.hidden_dim), torch.eye(self.hidden_dim))
        self.x_init_mvn = mvn_to_funsor(x_init_mvn, real_inputs=OrderedDict([('x_0', funsor.reals(self.hidden_dim))]))

    # we construct the various funsors used to compute the marginal log probability and other model quantities.
    # these funsors depend on the various model parameters.
    def get_tensors_and_dists(self):
        # normalize the transition probabilities
        trans_logits = self.transition_logits - self.transition_logits.logsumexp(dim=-1, keepdim=True)
        trans_probs = funsor.Tensor(trans_logits, OrderedDict([("s", funsor.bint(self.num_components))]))

        trans_mvn = torch.distributions.MultivariateNormal(torch.zeros(self.hidden_dim),
                                                           self.log_transition_noise.exp().diag_embed())
        obs_mvn = torch.distributions.MultivariateNormal(torch.zeros(self.obs_dim),
                                                         self.log_obs_noise.exp().diag_embed())

        event_dims = ("s",) if self.fine_transition_matrix or self.fine_transition_noise else ()
        x_trans_dist = matrix_and_mvn_to_funsor(self.transition_matrix, trans_mvn, event_dims, "x", "y")
        event_dims = ("s",) if self.fine_observation_matrix or self.fine_observation_noise else ()
        y_dist = matrix_and_mvn_to_funsor(self.observation_matrix, obs_mvn, event_dims, "x", "y")

        return trans_logits, trans_probs, trans_mvn, obs_mvn, x_trans_dist, y_dist

    # compute the marginal log probability of the observed data using a moment-matching approximation
    @funsor.interpreter.interpretation(funsor.terms.moment_matching)
    def log_prob(self, data):
        trans_logits, trans_probs, trans_mvn, obs_mvn, x_trans_dist, y_dist = self.get_tensors_and_dists()

        log_prob = funsor.Number(0.)

        s_vars = {-1: funsor.Tensor(torch.tensor(0), dtype=self.num_components)}
        x_vars = {}

        for t, y in enumerate(data):
            # construct free variables for s_t and x_t
            s_vars[t] = funsor.Variable(f's_{t}', funsor.bint(self.num_components))
            x_vars[t] = funsor.Variable(f'x_{t}', funsor.reals(self.hidden_dim))

            # incorporate the discrete switching dynamics
            log_prob += dist.Categorical(trans_probs(s=s_vars[t - 1]), value=s_vars[t])

            # incorporate the prior term p(x_t | x_{t-1})
            if t == 0:
                log_prob += self.x_init_mvn(value=x_vars[t])
            else:
                log_prob += x_trans_dist(s=s_vars[t], x=x_vars[t - 1], y=x_vars[t])

            # do a moment-matching reduction. at this point log_prob depends on (moment_matching_lag + 1)-many
            # pairs of free variables.
            if t > self.moment_matching_lag - 1:
                log_prob = log_prob.reduce(ops.logaddexp, frozenset([s_vars[t - self.moment_matching_lag].name,
                                                                     x_vars[t - self.moment_matching_lag].name]))

            # incorporate the observation p(y_t | x_t, s_t)
            log_prob += y_dist(s=s_vars[t], x=x_vars[t], y=y)

        T = data.shape[0]
        # reduce any remaining free variables
        for t in range(self.moment_matching_lag):
            log_prob = log_prob.reduce(ops.logaddexp, frozenset([s_vars[T - self.moment_matching_lag + t].name,
                                                                 x_vars[T - self.moment_matching_lag + t].name]))

        # assert that we've reduced all the free variables in log_prob
        assert not log_prob.inputs, 'unexpected free variables remain'

        # return the PyTorch tensor behind log_prob (which we can directly differentiate)
        return log_prob.data

    # do filtering, prediction, and smoothing using a moment-matching approximation.
    # here we implicitly use a moment matching lag of L = 1. the general logic follows
    # the logic in the log_prob method.
    @torch.no_grad()
    @funsor.interpreter.interpretation(funsor.terms.moment_matching)
    def filter_and_predict(self, data, smoothing=False):
        trans_logits, trans_probs, trans_mvn, obs_mvn, x_trans_dist, y_dist = self.get_tensors_and_dists()

        log_prob = funsor.Number(0.)

        s_vars = {-1: funsor.Tensor(torch.tensor(0), dtype=self.num_components)}
        x_vars = {-1: None}

        predictive_x_dists, predictive_y_dists, filtering_dists = [], [], []
        test_LLs = []

        for t, y in enumerate(data):
            s_vars[t] = funsor.Variable(f's_{t}', funsor.bint(self.num_components))
            x_vars[t] = funsor.Variable(f'x_{t}', funsor.reals(self.hidden_dim))

            log_prob += dist.Categorical(trans_probs(s=s_vars[t - 1]), value=s_vars[t])

            if t == 0:
                log_prob += self.x_init_mvn(value=x_vars[t])
            else:
                log_prob += x_trans_dist(s=s_vars[t], x=x_vars[t - 1], y=x_vars[t])

            if t > 0:
                log_prob = log_prob.reduce(ops.logaddexp, frozenset([s_vars[t - 1].name, x_vars[t - 1].name]))

            # do 1-step prediction and compute test LL
            if t > 0:
                predictive_x_dists.append(log_prob)
                _log_prob = log_prob - log_prob.reduce(ops.logaddexp)
                predictive_y_dist = y_dist(s=s_vars[t], x=x_vars[t]) + _log_prob
                test_LLs.append(predictive_y_dist(y=y).reduce(ops.logaddexp).data.item())
                predictive_y_dist = predictive_y_dist.reduce(ops.logaddexp, frozenset([f"x_{t}", f"s_{t}"]))
                predictive_y_dists.append(funsor_to_mvn(predictive_y_dist, 0, ()))

            log_prob += y_dist(s=s_vars[t], x=x_vars[t], y=y)

            # save filtering dists for forward-backward smoothing
            if smoothing:
                filtering_dists.append(log_prob)

        # do the backward recursion using previously computed ingredients
        if smoothing:
            # seed the backward recursion with the filtering distribution at t=T
            smoothing_dists = [filtering_dists[-1]]
            T = data.size(0)

            s_vars = {t: funsor.Variable(f's_{t}', funsor.bint(self.num_components)) for t in range(T)}
            x_vars = {t: funsor.Variable(f'x_{t}', funsor.reals(self.hidden_dim)) for t in range(T)}

            # do the backward recursion.
            # let p[t|t-1] be the predictive distribution at time step t.
            # let p[t|t] be the filtering distribution at time step t.
            # let f[t] denote the prior (transition) density at time step t.
            # then the smoothing distribution p[t|T] at time step t is
            # given by the following recursion.
            # p[t-1|T] = p[t-1|t-1] <p[t|T] f[t] / p[t|t-1]>
            # where <...> denotes integration of the latent variables at time step t.
            for t in reversed(range(T - 1)):
                integral = smoothing_dists[-1] - predictive_x_dists[t]
                integral += dist.Categorical(trans_probs(s=s_vars[t]), value=s_vars[t + 1])
                integral += x_trans_dist(s=s_vars[t], x=x_vars[t], y=x_vars[t + 1])
                integral = integral.reduce(ops.logaddexp, frozenset([s_vars[t + 1].name, x_vars[t + 1].name]))
                smoothing_dists.append(filtering_dists[t] + integral)

        # compute predictive test MSE and predictive variances
        predictive_means = torch.stack([d.mean for d in predictive_y_dists])  # T-1 ydim
        predictive_vars = torch.stack([d.covariance_matrix.diagonal(dim1=-1, dim2=-2) for d in predictive_y_dists])
        predictive_mse = (predictive_means - data[1:, :]).pow(2.0).mean(-1)

        if smoothing:
            # compute smoothed mean function
            smoothing_dists = [funsor_to_cat_and_mvn(d, 0, (f"s_{t}",))
                               for t, d in enumerate(reversed(smoothing_dists))]
            means = torch.stack([d[1].mean for d in smoothing_dists])  # T 2 xdim
            means = torch.matmul(means.unsqueeze(-2), self.observation_matrix).squeeze(-2)  # T 2 ydim

            probs = torch.stack([d[0].logits for d in smoothing_dists]).exp()
            probs = probs / probs.sum(-1, keepdim=True)  # T 2

            smoothing_means = (probs.unsqueeze(-1) * means).sum(-2)  # T ydim
            smoothing_probs = probs[:, 1]

            return predictive_mse, torch.tensor(np.array(test_LLs)), predictive_means, predictive_vars, \
                smoothing_means, smoothing_probs
        else:
            return predictive_mse, torch.tensor(np.array(test_LLs))


def main(args):
    # download and pre-process EEG data if not in test mode
    if not args.test:
        download_data()
        N_val, N_test = 149, 200
        data = np.loadtxt('eeg.dat', delimiter=',', skiprows=19)
        print("[raw data shape] {}".format(data.shape))
        data = data[::20, :]
        print("[data shape after thinning] {}".format(data.shape))
        eye_state = [int(l) for l in data[:, -1].tolist()]
        data = torch.tensor(data[:, :-1]).float()
    # in test mode (for continuous integration on github) so create fake data
    else:
        data = torch.randn(10, 3)
        N_val, N_test = 2, 2

    T, obs_dim = data.shape
    N_train = T - N_test - N_val

    np.random.seed(0)
    rand_perm = np.random.permutation(N_val + N_test)
    val_indices = rand_perm[0:N_val]
    test_indices = rand_perm[N_val:]

    data_mean = data[0:N_train, :].mean(0)
    data -= data_mean
    data_std = data[0:N_train, :].std(0)
    data /= data_std

    print("Length of time series T: {}   Observation dimension: {}".format(T, obs_dim))
    print("N_train: {}  N_val: {}  N_test: {}".format(N_train, N_val, N_test))

    torch.manual_seed(args.seed)

    # set up model
    slds = SLDS(num_components=args.num_components, hidden_dim=args.hidden_dim, obs_dim=obs_dim,
                fine_observation_noise=args.fon, fine_transition_noise=args.ftn,
                fine_observation_matrix=args.fom, fine_transition_matrix=args.ftm,
                moment_matching_lag=args.moment_matching_lag)

    # set up optimizer
    adam = torch.optim.Adam(slds.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999), amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(adam, gamma=args.gamma)
    ts = [time.time()]

    report_frequency = 1

    # training loop
    for step in range(args.num_steps):
        nll = -slds.log_prob(data[0:N_train, :]) / N_train
        nll.backward()

        if step == 5:
            scheduler.base_lrs[0] *= 0.20

        adam.step()
        scheduler.step()
        adam.zero_grad()

        if step % report_frequency == 0 or step == args.num_steps - 1:
            step_dt = ts[-1] - ts[-2] if step > 0 else 0.0
            pred_mse, pred_LLs = slds.filter_and_predict(data[0:N_train + N_val + N_test, :])
            val_mse = pred_mse[val_indices].mean().item()
            test_mse = pred_mse[test_indices].mean().item()
            val_ll = pred_LLs[val_indices].mean().item()
            test_ll = pred_LLs[test_indices].mean().item()

            stats = "[step %03d] train_nll: %.5f val_mse: %.5f val_ll: %.5f test_mse: %.5f test_ll: %.5f\t(dt: %.2f)"
            print(stats % (step, nll.item(), val_mse, val_ll, test_mse, test_ll, step_dt))

        ts.append(time.time())

    # plot predictions and smoothed means
    if args.plot:
        assert not args.test
        predicted_mse, LLs, pred_means, pred_vars, smooth_means, smooth_probs = \
            slds.filter_and_predict(data, smoothing=True)

        pred_means = pred_means.data.numpy()
        pred_stds = pred_vars.sqrt().data.numpy()
        smooth_means = smooth_means.data.numpy()
        smooth_probs = smooth_probs.data.numpy()

        import matplotlib
        matplotlib.use('Agg')  # noqa: E402
        import matplotlib.pyplot as plt

        f, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        T = data.size(0)
        N_valtest = N_val + N_test
        to_seconds = 117.0 / T

        for k, ax in enumerate(axes[:-1]):
            which = [0, 4, 10][k]
            ax.plot(to_seconds * np.arange(T), data[:, which], 'ko', markersize=2)
            ax.plot(to_seconds * np.arange(N_train), smooth_means[:N_train, which], ls='solid', color='r')

            ax.plot(to_seconds * (N_train + np.arange(N_valtest)),
                    pred_means[-N_valtest:, which], ls='solid', color='b')
            ax.fill_between(to_seconds * (N_train + np.arange(N_valtest)),
                            pred_means[-N_valtest:, which] - 1.645 * pred_stds[-N_valtest:, which],
                            pred_means[-N_valtest:, which] + 1.645 * pred_stds[-N_valtest:, which],
                            color='lightblue')
            ax.set_ylabel("$y_{%d}$" % (which + 1), fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=14)

        axes[-1].plot(to_seconds * np.arange(T), eye_state, 'k', ls='solid')
        axes[-1].plot(to_seconds * np.arange(T), smooth_probs, 'r', ls='solid')
        axes[-1].set_xlabel("Time (s)", fontsize=20)
        axes[-1].set_ylabel("Eye state", fontsize=20)
        axes[-1].tick_params(axis='both', which='major', labelsize=14)

        plt.tight_layout(pad=0.7)
        plt.savefig('eeg.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Switching linear dynamical system")
    parser.add_argument("-n", "--num-steps", default=3, type=int)
    parser.add_argument("-s", "--seed", default=15, type=int)
    parser.add_argument("-hd", "--hidden-dim", default=5, type=int)
    parser.add_argument("-k", "--num-components", default=2, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.5, type=float)
    parser.add_argument("-b1", "--beta1", default=0.75, type=float)
    parser.add_argument("-g", "--gamma", default=0.99, type=float)
    parser.add_argument("-mml", "--moment-matching-lag", default=1, type=int)
    parser.add_argument("--plot", action='store_true')
    parser.add_argument("--fon", action='store_true')
    parser.add_argument("--ftm", action='store_true')
    parser.add_argument("--fom", action='store_true')
    parser.add_argument("--ftn", action='store_true')
    parser.add_argument("--test", action='store_true')
    args = parser.parse_args()

    main(args)
