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
from funsor.domains import bint, reals
from funsor.torch import Tensor, Variable
from funsor.gaussian import Gaussian


def generate_data(num_frames, num_sensors):
    # simulate biased sensors
    sensors = []
    full_observations = []
    for _ in range(num_sensors):
        bias = 1. * torch.randn(2)
        sensors.append(bias)

    # simulate all sensor observations
    z = 10 * torch.rand(2)  # initial state
    v = torch.randn(2)  # velocity
#     f = torch.randn(2, 2)
#     h = torch.randn(2, 2)
#     Q = torch.eye(2)
#     R = torch.eye(2)

    for t in range(num_frames):
        # Advance latent state.
        z += v + 0.1 * torch.randn(2)
        x = z.expand([num_sensors, 2]) - torch.stack(sensors)
#     z += f @ z + dist.MultivariateNormal(torch.zeros(2), Q)
#     y += h @ z + dist.MultivariateNormal(torch.zeros(2), R)
        full_observations.append(x)
    full_observations = torch.stack(full_observations)
    assert full_observations.shape == (num_frames, num_sensors, 2)
    return full_observations, sensors


class HMM(nn.Module):
    def __init__(self, num_sensors, state_dim=2):
        super(HMM, self).__init__()
        self.num_sensors = num_sensors
        self.state_dim = state_dim

        # learnable params
        self.bias_scales = nn.Parameter(torch.ones(2))
        self.obs_noise = nn.Parameter(torch.tensor(1.))
        self.trans_noise = nn.Parameter(torch.tensor(1.))

    def forward(self, track, add_bias=True):
        num_sensors = self.num_sensors
        obs_dim = num_sensors * self.state_dim

        # reshape the data
#         assert isinstance(track, Tensor)
#         data = Tensor(track.data.reshape(-1, obs_dim),
#                       OrderedDict(time=bint(len(track.data))))

        # bias distribution
        bias = Variable('bias', reals(obs_dim))
        assert not torch.isnan(self.bias_scales).any(), "bias scales was nan"
        bias_dist = dist_to_funsor(
            dist.MultivariateNormal(
                torch.zeros(num_sensors * 2),
                self.bias_scales.expand(num_sensors, 2).reshape(-1).diag_embed()
            )
        )(value=bias)

        # this needs to be a funsor dist
        init_dist = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2) + 0.1 * torch.randn(2))
        self.init = dist_to_funsor(init_dist)(value="state")

        # hidden states
        prev = Variable("prev", reals(self.state_dim))
        curr = Variable("curr", reals(self.state_dim))
        # inputs are the previous state ``state`` and the next state
        # ncv transition matrix todo
#         transition_matrix = 0.1 * torch.randn(state_dim, state_dim) + self.transition_param.diag_embed().flip(0)
        transition_matrix = torch.randn(self.state_dim, self.state_dim)
        # ncv transition noise todo
        trans_noise = self.trans_noise.expand([self.state_dim]).diag_embed()
        self.trans_dist = f_dist.MultivariateNormal(
            loc=prev @ transition_matrix,
            scale_tril=trans_noise,
            value=curr
            )

        # free variables that have distributions over them
        state = Variable('state', reals(self.state_dim))
        obs = Variable("obs", reals(obs_dim))
        # observation
        observation_matrix = Tensor(torch.eye(self.state_dim, self.state_dim).expand(num_sensors, -1, -1).
                transpose(0, -1).reshape(self.state_dim, obs_dim))
        assert observation_matrix.output.shape == (self.state_dim, obs_dim), observation_matrix.output.shape
        obs_noise = self.obs_noise.expand(obs_dim).diag_embed()
        obs_loc = state @ observation_matrix
        if add_bias:
            obs_loc += bias
        self.observation_dist = f_dist.MultivariateNormal(
            loc=obs_loc,
#             loc=Tensor(torch.zeros(obs_dim)),
#             scale_tril=Tensor(torch.eye(obs_dim)),
            scale_tril=Tensor(obs_noise),
            value=obs
        )

        logp = bias_dist
        curr = "state_init"
        logp += self.init(state=curr)
#         import pdb; pdb.set_trace()
        for t, x in enumerate(track):
            x = x.expand([num_sensors, -1]).reshape(-1)
            prev, curr = curr, f"state_{t}"
            # transition to state at t=1
            logp += self.trans_dist(prev=prev, curr=curr)

            logp += self.observation_dist(state=curr, obs=x)
            logp = logp.reduce(ops.logaddexp, prev)
        # marginalize out remaining latent variables
        # use mvn_to_funsor to pull out bias cov
        # plot trace or max e-value
        cov = logp.terms[1].precision
        logp = logp.reduce(ops.logaddexp)

        # we should get a single scalar Tensor here
        assert isinstance(logp, Tensor) and logp.data.dim() == 0, logp.pretty()
        return logp.data, cov


def main(args):
    print(f'running with bias={not args.no_bias}')
    torch.manual_seed(12)
    losses = []
    # params.append(transition_matrix)
    model = HMM(args.num_sensors)
    optim = Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 200, gamma=0.2)
    data, biases = generate_data(args.frames[-1], args.num_sensors)
    for f in args.frames:
        print(f'running data with {f} frames')
        # must do this since funsor slicing not supported
        truncated_data = data[:f]
        for i in range(args.num_epochs):
            optim.zero_grad()
            log_prob, cov = model(truncated_data, add_bias=not args.no_bias)
            loss = -log_prob
            loss.backward()
            losses.append(loss.item())
            if i % 10 == 0:
                print(loss.item())
            optim.step()
            scheduler.step()
        md = {
#                 "bias_locs": model.bias_locs,
                "bias_scales": model.bias_scales,
                "losses": losses,
                "data": data.data,
                "biases": biases,
                "cov": cov
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
