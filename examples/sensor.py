import argparse
import itertools
import math
import os

import pyro.distributions as dist
import torch
import torch.nn as nn
from torch.optim import Adam

import funsor.distributions as f_dist
import funsor.ops as ops
from funsor.domains import reals
from funsor.pyro.convert import dist_to_funsor, funsor_to_mvn
from funsor.torch import Tensor, Variable

# We use a 2D continuous-time NCV dynamics model throughout.
# See http://webee.technion.ac.il/people/shimkin/Estimation09/ch8_target.pdf
TIME_STEP = 1.
NCV_PROCESS_NOISE = torch.tensor([[1/3, 0.0, 1/2, 0.0],
                                  [0.0, 1/3, 0.0, 1/2],
                                  [1/2, 0.0, 1.0, 0.0],
                                  [0.0, 1/2, 0.0, 1.0]])
NCV_TRANSITION_MATRIX = torch.tensor([[1., 0., 0., 0.],
                                      [0., 1., 0., 0.],
                                      [1., 0., 1., 0.],
                                      [0., 1., 0., 1.]])


@torch.no_grad()
def generate_data(num_frames, num_sensors):
    """
    Generate data from a damped NCV dynamics model
    """
    dt = TIME_STEP
    bias_scale = 3.0
    process_noise = 0.1
    obs_noise = 1.0

    # define dynamics
    z = torch.cat([10. * torch.randn(2),  # position
                   0.1 * torch.rand(2)])  # velocity
    damp = 0.5  # damp the velocities
    f = torch.tensor([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [dt * math.exp(-damp * dt), 0, math.exp(-damp * dt), 0],
                      [0, dt * math.exp(-damp * dt), 0, math.exp(-damp * dt)]])
    Q = process_noise * NCV_PROCESS_NOISE
    trans_dist = dist.MultivariateNormal(torch.zeros(4), Q)

    # define biased sensors
    sensor_bias = bias_scale * torch.randn(2, num_sensors)
    h = torch.eye(4, 2).unsqueeze(-1).expand(-1, -1, num_sensors).reshape(4, -1)
    R = obs_noise * torch.eye(2 * num_sensors)
    obs_dist = dist.MultivariateNormal(sensor_bias.reshape(-1), R)

    states = []
    observations = []
    for t in range(num_frames):
        z = z @ f + trans_dist.sample()
        states.append(z)

        x = z @ h + obs_dist.sample()
        observations.append(x)

    states = torch.stack(states)
    observations = torch.stack(observations)
    assert observations.shape == (num_frames, num_sensors * 2)
    return observations, states, sensor_bias


class HMM(nn.Module):
    def __init__(self, num_sensors):
        super(HMM, self).__init__()
        self.num_sensors = num_sensors

        # learnable params
        self.log_bias_scale = nn.Parameter(torch.tensor(0.))
        self.log_obs_noise = nn.Parameter(torch.tensor(0.))
        self.log_trans_noise = nn.Parameter(torch.tensor(0.))

    def forward(self, observations, add_bias=True):
        obs_dim = 2 * self.num_sensors
        bias_scale = self.log_bias_scale.exp()
        obs_noise = self.log_obs_noise.exp()
        trans_noise = self.log_trans_noise.exp()

        # bias distribution
        bias = Variable('bias', reals(obs_dim))
        assert not torch.isnan(bias_scale), "bias scales was nan"
        bias_dist = dist_to_funsor(
            dist.MultivariateNormal(
                torch.zeros(obs_dim),
                scale_tril=bias_scale.expand(2 * self.num_sensors).diag_embed()
            )
        )(value=bias)

        init_dist = torch.distributions.MultivariateNormal(
            torch.zeros(4), scale_tril=10. * torch.eye(4))
        self.init = dist_to_funsor(init_dist)(value="state")

        # hidden states
        prev = Variable("prev", reals(4))
        curr = Variable("curr", reals(4))
        self.trans_dist = f_dist.MultivariateNormal(
            loc=prev @ NCV_TRANSITION_MATRIX,
            scale_tril=trans_noise * NCV_PROCESS_NOISE.cholesky(),
            value=curr
            )

        state = Variable('state', reals(4))
        obs = Variable("obs", reals(obs_dim))
        observation_matrix = Tensor(torch.eye(4, 2).unsqueeze(-1)
                                    .expand(-1, -1, self.num_sensors).reshape(4, -1))
        assert observation_matrix.output.shape == (4, obs_dim), observation_matrix.output.shape
        obs_noise = obs_noise.expand(obs_dim).diag_embed()
        obs_loc = state @ observation_matrix
        if add_bias:
            obs_loc += bias
        self.observation_dist = f_dist.MultivariateNormal(
            loc=obs_loc,
            scale_tril=obs_noise,
            value=obs
        )

        logp = bias_dist
        curr = "state_init"
        logp += self.init(state=curr)
        for t, x in enumerate(observations):
            prev, curr = curr, f"state_{t}"
            logp += self.trans_dist(prev=prev, curr=curr)
            logp += self.observation_dist(state=curr, obs=x)
            # marginalize out previous state
            logp = logp.reduce(ops.logaddexp, prev)
        # marginalize out bias variable
        logp = logp.reduce(ops.logaddexp, "bias")

        # save posterior over the final state
        assert set(logp.inputs) == {f'state_{len(observations) - 1}'}
        posterior = funsor_to_mvn(logp, ndims=0)

        # marginalize out remaining variables
        logp = logp.reduce(ops.logaddexp)
        assert isinstance(logp, Tensor) and logp.shape == (), logp.pretty()
        return logp.data, posterior


def track(args):
    results = {}  # keyed on (seed, bias, num_frames)
    for seed in args.seed:
        torch.manual_seed(seed)
        observations, states, sensor_bias = generate_data(max(args.num_frames), args.num_sensors)
        for bias, num_frames in itertools.product(args.bias, args.num_frames):
            print(f'tracking with seed={seed}, bias={bias}, num_frames={num_frames}')
            model = HMM(args.num_sensors)
            optim = Adam(model.parameters(), lr=args.lr)
            losses = []
            for i in range(args.num_epochs):
                optim.zero_grad()
                log_prob, posterior = model(observations[:num_frames], add_bias=bias)
                loss = -log_prob
                loss.backward()
                losses.append(loss.item())
                if i % 10 == 0:
                    print(loss.item())
                optim.step()

            # Collect evaluation metrics.
            final_state_true = states[num_frames - 1]
            assert final_state_true.shape == (4,)
            final_pos_true = final_state_true[:2]
            final_vel_true = final_state_true[2:]

            final_state_est = posterior.loc
            assert final_state_est.shape == (4,)
            final_pos_est = final_state_est[:2]
            final_vel_est = final_state_est[2:]
            final_pos_error = float(torch.norm(final_pos_true - final_pos_est))
            final_vel_error = float(torch.norm(final_vel_true - final_vel_est))
            print(f'final_pos_error = {final_pos_error}')

            results[seed, bias, num_frames] = {
                "args": args,
                "observations": observations[:num_frames],
                "states": states[:num_frames],
                "sensor_bias": sensor_bias,
                "losses": losses,
                "bias_scale": float(model.log_bias_scale.exp()),
                "obs_noise": float(model.log_obs_noise.exp()),
                "trans_noise": float(model.log_trans_noise.exp()),
                "final_state_estimate": posterior,
                "final_pos_error": final_pos_error,
                "final_vel_error": final_vel_error,
            }
        print(f'saving output to: {args.metrics_filename}')
        torch.save(results, args.metrics_filename)


def main(args):
    if args.force or not os.path.exists(args.metrics_filename):
        track(args)

    if args.plot_filename:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot
        import numpy as np
        results = torch.load(args.metrics_filename)
        seeds = set(seed for seed, _, _ in results)
        X = args.num_frames
        pyplot.figure(figsize=(5, 1.5), dpi=300)
        pyplot.plot(X, [np.mean([results[s, 0, f]['final_pos_error']**2 for s in seeds])
                        for f in args.num_frames], 'k--')
        pyplot.plot(X, [np.mean([results[s, 1, f]['final_pos_error']**2 for s in seeds])
                        for f in args.num_frames], 'r-')
        pyplot.ylabel('Position RMSE')
        pyplot.xlabel('Track Length')
        pyplot.xticks((2, 5, 10, 15, 20))
        pyplot.xlim(2, 20)
        pyplot.tight_layout(0)
        pyplot.savefig(args.plot_filename)


def int_list(arg):
    return [int(n) for n in arg.split(',')]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Biased Kalman filter")
    parser.add_argument("--seed", default="0,1,2,3,4,5,6,7,8,9", type=int_list,
                        help="random seed, comma delimited for multiple runs")
    parser.add_argument("--bias", default="0,1", type=int_list,
                        help="whether to model bias, comma deliminted for multiple runs")
    parser.add_argument("-f", "--num-frames", default="2,4,6,8,10,12,14,16,18,20",
                        type=int_list,
                        help="number of sensor frames, comma delimited for multiple runs")
    parser.add_argument("--num-sensors", default=5, type=int)
    parser.add_argument("-n", "--num-epochs", default=100, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--metrics-filename", default="sensor.pkl", type=str)
    parser.add_argument("--plot-filename", default="", type=str)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args)
