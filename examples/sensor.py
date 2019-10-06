import argparse

import math

import torch
import torch.nn as nn
from torch.optim import Adam

import pyro.distributions as dist

import funsor.distributions as f_dist
import funsor.ops as ops
from funsor.pyro.convert import dist_to_funsor
from funsor.domains import reals
from funsor.torch import Tensor, Variable


NCV_PROCESS_NOISE = torch.tensor([[1./3., 0., 0.5, 0.],
                                  [0., 1./3., 0., 0.5],
                                  [0.5, 0., 1., 0.],
                                  [0., 0.5, 0., 1.]])


NCV_TRANSITION_MATRIX = torch.tensor([[1., 0., 1., 0.],
                                      [0., 1., 0., 1.],
                                      [0., 0., 1., 0.],
                                      [0., 0., 0., 1.]])


def generate_data(num_frames, num_sensors):
    """
    Generate data from an NCV dynamics model
    """
    dt = 1
    sensors = []
    full_observations = []
    # simulate biased sensors
    for _ in range(num_sensors):
        bias = 3 * torch.randn(2)
        sensors.append(bias)

    z = torch.cat([torch.zeros(2), 0.1 * torch.rand(2)]).unsqueeze(1)  # PV vector
    # damp the velocities
    damp = 0.5
    f = torch.tensor([[1, 0, dt * math.exp(-damp * dt), 0],
                      [0, 1, 0, dt * math.exp(-damp * dt)],
                      [0, 0, math.exp(-damp * dt), 0],
                      [0, 0, 0, math.exp(-damp * dt)]])
    h = torch.eye(2, 4)
    R = torch.eye(2)

    for t in range(num_frames):
        z = f @ z + dist.MultivariateNormal(torch.zeros(4), NCV_PROCESS_NOISE).sample().unsqueeze(1)
        x = h @ z + dist.MultivariateNormal(torch.zeros(2), R).sample().unsqueeze(1)
        x = x.transpose(0, 1).expand([num_sensors, 2]) - torch.stack(sensors)
        full_observations.append(x.clone())
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
        obs_dim = self.num_sensors * 2

        # bias distribution
        bias = Variable('bias', reals(obs_dim))
        assert not torch.isnan(self.bias_scales).any(), "bias scales was nan"
        bias_dist = dist_to_funsor(
            dist.MultivariateNormal(
                torch.zeros(obs_dim),
                self.bias_scales.expand(self.num_sensors, 2).reshape(-1).diag_embed()
            )
        )(value=bias)

        init_dist = torch.distributions.MultivariateNormal(torch.zeros(4), torch.eye(4))
        self.init = dist_to_funsor(init_dist)(value="state")

        # hidden states
        prev = Variable("prev", reals(4))
        curr = Variable("curr", reals(4))
        trans_noise = self.trans_noise * NCV_PROCESS_NOISE.cholesky()
        self.trans_dist = f_dist.MultivariateNormal(
            loc=prev @ NCV_TRANSITION_MATRIX,
            scale_tril=trans_noise,
            value=curr
            )

        state = Variable('state', reals(4))
        obs = Variable("obs", reals(obs_dim))
        observation_matrix = Tensor(torch.eye(4, 2).expand(self.num_sensors, -1, -1).
                                    transpose(0, -1).reshape(4, obs_dim))
        assert observation_matrix.output.shape == (4, obs_dim), observation_matrix.output.shape
        obs_noise = self.obs_noise.expand(obs_dim).diag_embed()
        obs_loc = state @ observation_matrix
        if add_bias:
            obs_loc += bias
        self.observation_dist = f_dist.MultivariateNormal(
            loc=obs_loc,
            scale_tril=Tensor(obs_noise),
            value=obs
        )

        logp = bias_dist
        curr = "state_init"
        logp += self.init(state=curr)
        for t, x in enumerate(track):
            x = x.expand([self.num_sensors, -1]).reshape(-1)
            prev, curr = curr, f"state_{t}"
            # transition to state at t=1
            logp += self.trans_dist(prev=prev, curr=curr)

            logp += self.observation_dist(state=curr, obs=x)
            logp = logp.reduce(ops.logaddexp, prev)
        # marginalize out remaining latent variables
        prec = logp.terms[1].precision
        logp = logp.reduce(ops.logaddexp)

        # we should get a single scalar Tensor here
        assert isinstance(logp, Tensor) and logp.data.dim() == 0, logp.pretty()
        return logp.data, prec


def main(args):
    print(f'running with bias={not args.no_bias}')
    torch.manual_seed(12)
    data, biases = generate_data(args.frames[-1], args.num_sensors)
    for f in args.frames:
        losses = []
        model = HMM(args.num_sensors)
        optim = Adam(model.parameters(), lr=0.1)
        print(f'running data with {f} frames')
        truncated_data = data[:f]
        for i in range(args.num_epochs):
            optim.zero_grad()
            log_prob, prec = model(truncated_data, add_bias=not args.no_bias)
            loss = -log_prob
            loss.backward()
            losses.append(loss.item())
            if i % 10 == 0:
                print(loss.item())
            optim.step()
        md = {
                "bias_scales": model.bias_scales,
                "losses": losses,
                "data": data.data,
                "biases": biases,
                "prec": prec
             }
        suffix = f'_{f}_bias={not args.no_bias}.pkl'
        print(f'saving output to: {args.save}' + suffix)
        torch.save(md, args.save + suffix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Switching linear dynamical system")
    parser.add_argument("-n", "--num-epochs", default=300, type=int)
    parser.add_argument("--no-bias", default=False, action="store_true")
    parser.add_argument("--frames", default="200", type=lambda s: [int(i) for i in s.split(',')],
                        help="frames to run, comma delimited")
    parser.add_argument("--save", default="sensor", type=str)
    parser.add_argument("--num-sensors", default=5, type=int)
    parser.add_argument("--plot", default=False, action="store_true")
    args = parser.parse_args()
    main(args)
