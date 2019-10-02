import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import Adam

import pyro.distributions as dist

import funsor
import funsor.pyro
import funsor.distributions as f_dist
import funsor.ops as ops
from funsor.pyro.convert import dist_to_funsor, mvn_to_funsor, matrix_and_mvn_to_funsor, tensor_to_funsor
from funsor.interpreter import interpretation, reinterpret
from funsor.optimizer import apply_optimizer
from funsor.terms import eager
from funsor.domains import bint, reals
from funsor.torch import Tensor, Variable
from funsor.gaussian import Gaussian
from funsor.sum_product import sequential_sum_product


def generate_data(num_frames, num_sensors):
    # simulate biased sensors
    sensors = []
    full_observations = []
    for _ in range(num_sensors):
        bias = 5. * torch.randn(2)
        sensors.append(bias)

    # simulate all sensor observations
    z = 10 * torch.rand(2)  # initial state
    v = torch.randn(2)  # velocity
    for t in range(num_frames):
        # Advance latent state.
        z += v + 0.1 * torch.randn(2)
        x = z.expand([num_sensors, 2]) - torch.stack(sensors)
        full_observations.append(x)
    full_observations = torch.stack(full_observations)
    assert full_observations.shape == (num_frames, num_sensors, 2)
    full_observations = Tensor(full_observations)["time"]
    return full_observations, sensors


class HMM(nn.Module):
    def __init__(self, num_sensors, state_dim):
        super(HMM, self).__init__()
        self.num_sensors = num_sensors
        self.state_dim = state_dim

        # learnable params
        self.bias_locs = nn.Parameter(torch.zeros(2))
        self.bias_scales = nn.Parameter(torch.ones(2))
        self.transition_param = nn.Parameter(torch.ones(state_dim))

        self.obs_noise = nn.Parameter(torch.eye(state_dim, self.num_sensors * 2))

        self.trans_noise = nn.Parameter(torch.ones(1))

    def forward(self, track, add_bias=True):
        num_sensors = self.num_sensors
        obs_dim = num_sensors * 2
        state_dim = self.state_dim

        # reshape the data
        assert isinstance(track, Tensor)
        data = Tensor(track.data.reshape(-1, obs_dim),
                      OrderedDict(time=bint(len(track.data))))

        # bias distribution
        bias = Variable('bias', reals(obs_dim))
        assert not torch.isnan(self.bias_locs).any(), "bias locs was nan"
        assert not torch.isnan(self.bias_scales).any(), "bias scales was nan"
#         bias_scales = self.bias_scales.expand(num_sensors, 2).reshape(-1).diag_embed()
        bias_dist = dist_to_funsor(
            dist.MultivariateNormal(
#                 torch.zeros(num_sensors * 2),
                self.bias_locs.expand(num_sensors, 2).reshape(-1).diag_embed(),
                self.bias_scales.expand(num_sensors, 2).reshape(-1).diag_embed()
            )
        )(value=bias)

        # this needs to be a funsor dist
        init_dist = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2) + 0.1 * torch.randn(2))
        self.init = dist_to_funsor(init_dist)(value="state")

        # hidden states
        prev = Variable("prev", reals(state_dim))
        curr = Variable("curr", reals(state_dim))
        # inputs are the previous state ``state`` and the next state
        transition_matrix = 0.1 * torch.randn(state_dim, state_dim) + self.transition_param.diag_embed().flip(0)
        trans_noise = 0.1 * torch.randn(state_dim, state_dim) + self.trans_noise.expand([state_dim]).diag_embed()
        self.trans_dist = f_dist.MultivariateNormal(
            loc=Tensor(torch.randn(state_dim)),
            scale_tril=Tensor(trans_noise),
            value=curr - prev @ Tensor(transition_matrix))

        # free variables that have distributions over them
        state = Variable('state', reals(state_dim))
        obs = Variable("obs", reals(obs_dim))
        # observation
        observation_matrix = Tensor(torch.randn(state_dim, obs_dim))
        value = value = state @ observation_matrix - obs  # this takes us from state to biased obs
        if add_bias:
            value += bias
        self.observation_dist = f_dist.MultivariateNormal(
            loc=Tensor(torch.zeros(obs_dim)),
            scale_tril=Tensor(self.obs_noise + 0.1 * torch.randn(obs_dim, obs_dim)),
            value=value
        )

        with interpretation(eager):
            logp = bias_dist
            logp += self.init
            state_0 = Variable("state_0", reals(state_dim))
            # observation at t=0
            logp += self.observation_dist(state=state_0, obs=data(time=0))
            state_1 = Variable("state_1", reals(state_dim))
            # transition to state at t=1
            logp += self.trans_dist(prev=state_0, curr=state_1)
            # observation at t=1
            logp += self.observation_dist(state=state_1, obs=data(time=1))
            # marginalize out remaining latent variables
            logp = logp.reduce(ops.logaddexp)

        # we should get a single scalar Tensor here
        assert isinstance(logp, Tensor) and logp.data.dim() == 0, logp.pretty()
        return logp.data

def plot():
    import matplotlib.pyplot as plt
    pass


def main(args):
    print(f'running with bias={not args.no_bias}')
    torch.manual_seed(12)
    losses = []
    # params.append(transition_matrix)
    model = HMM(args.num_sensors, args.num_sensors*2)
    optim = Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 100, gamma=0.2)
    data, biases = generate_data(args.frames[-1], args.num_sensors)
    for f in args.frames:
        print(f'running data with {f} frames')
        # must do this since funsor slicing not supported
        truncated_data = Tensor(data.data[:f])['time']
        for i in range(args.num_epochs):
            optim.zero_grad()
            log_prob = model(truncated_data, add_bias=not args.no_bias)
            loss = -log_prob
            loss.backward()
            losses.append(loss.item())
            if i % 10 == 0:
                print(loss.item())
            optim.step()
            scheduler.step()
        md = {
                "bias_locs": model.bias_locs,
                "bias_scales": model.bias_scales,
                "losses": losses,
                "data": data.data,
                "biases": biases
             }
        print(f'saving output to: {f}_{args.save}')
        torch.save(md, f'{f}_' + args.save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Switching linear dynamical system")
    parser.add_argument("-n", "--num-epochs", default=300, type=int)
    parser.add_argument("--no-bias", default=False, action="store_true")
    parser.add_argument("--frames", default="200", type=lambda s: [int(i) for i in s.split(',')],
                        help="frames to run, comma delimited")
    parser.add_argument("--save", default="sensor.pkl", type=str)
    parser.add_argument("--num-sensors", default=5, type=int)
    parser.add_argument("--plot", default=False, action="store_true")
    args = parser.parse_args()
    main(args)
