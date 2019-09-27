import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import Adam

import pyro
import pyro.distributions as dist

import funsor
import funsor.pyro
import funsor.distributions as f_dist
import funsor.ops as ops
from funsor.pyro.convert import dist_to_funsor, mvn_to_funsor, matrix_and_mvn_to_funsor, tensor_to_funsor
from funsor.interpreter import interpretation, reinterpret
from funsor.optimizer import apply_optimizer
from funsor.terms import lazy, eager_or_die
from funsor.domains import bint, reals
from funsor.torch import Tensor, Variable
from funsor.gaussian import Gaussian
from funsor.sum_product import sequential_sum_product

import matplotlib.pyplot as plt

num_sensors = 5
num_frames = 100

def generate_data():
    # simulate biased sensors
    sensors  = []
    for _ in range(num_sensors):
        bias = 0.5 * torch.randn(2)
        sensors.append(bias)

    # simulate a single track
    # TODO heterogeneous time
    partial_obs = []
    z = 10 * torch.rand(2)  # initial state
    v = 2 * torch.randn(2)  # velocity
    for t in range(num_frames):
        # Advance latent state.
        z += v + 0.1 * torch.randn(2)
    #     z.clamp_(min=0, max=10)  # keep in the box

        # Observe via a random sensor.
        sensor_id = pyro.sample('id', dist.Categorical(torch.ones(num_sensors)))
        x = z - sensors[sensor_id]
        partial_obs.append({"sensor_id": sensor_id, "x": x})

    # simulate all tracks
    full_observations = []
    z = 10 * torch.rand(5, 2)  # initial state
    v = 2 * torch.randn(5, 2)  # velocity
    for t in range(num_frames):
        # Advance latent state.
        z += v + 0.1 * torch.randn(5, 2)
    #     z.clamp_(min=0, max=10)  # keep in the box

        x = z - torch.stack(sensors)
        full_observations.append(x)
    full_observations = torch.stack(full_observations)
    assert full_observations.shape == (num_frames, 5, 2)
    full_observations = Tensor(full_observations)["time"]
    return full_observations

# TODO transform this to cholesky decomposition
# print(bias_cov.shape)
# bias_cov = bias_cov @ bias_cov.t()
# create a joint Gaussian over biases

# covs = [torch.eye(2, requires_grad=True) for i in range(num_sensors)]
# bias_dist = 0.
# for i in range(num_sensors):
#     bias += funsor.pyro.convert.mvn_to_funsor(
#         dist.MultivariateNormal(torch.zeros(2), covs[i]),
# #         event_dims=("pos",),
# #         real_inputs=OrderedDict([("bias_{}".format(i), reals(2))])
#         real_inputs=OrderedDict([("bias", reals(2))])
#     )(value="bias_{}".format(i))
# bias_dist.__dict__

# we can't write bias_dist as a sum of mvns because affine transformation
# of mvns is not supported yet.  instead we will combine all the sensors
# into a giant tensor
# bias_scales = torch.ones(2, requires_grad=True)  # This can be learned
# bias_dist = funsor.pyro.convert.mvn_to_funsor(
#     dist.MultivariateNormal(
#         torch.zeros(num_sensors * 2),
#         bias_scales.expand(num_sensors, 2).reshape(-1).diag_embed()
#     ),
#     real_inputs=OrderedDict([("bias", reals(num_sensors, 2))])
# )
# bias_dist.__dict__

class HMM(nn.Module):
    def __init__(self, num_sensors):
        super(HMM, self).__init__()
        self.num_sensors = num_sensors

    def forward(self, track):
        num_sensors = self.num_sensors

        bias_dist = funsor.pyro.convert.mvn_to_funsor(
            dist.MultivariateNormal(
                torch.zeros(num_sensors * 2),
                bias_scales.expand(num_sensors, 2).reshape(-1).diag_embed()
            ),
            real_inputs=OrderedDict([("bias", reals(num_sensors * 2))])
        )
        init_dist = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
        transition_dist = f_dist.MultivariateNormal(
            torch.zeros(2), trans_dist_cov)
        observation_matrix = torch.eye(2) + 0.2 * torch.randn(2, 2)
        observation_matrix = (observation_matrix.unsqueeze(1)
                                                .expand(2, num_sensors, 2)
                                                .reshape(2, num_sensors * 2))

        bias = torch.zeros(num_sensors * 2, requires_grad=True)
        # obs_dist will be flattened out
        observation_dist = f_dist.MultivariateNormal(
            bias,
            torch.eye(num_sensors * 2))

        self.init = dist_to_funsor(init_dist)(value="state")
        # inputs are the previous state ``state`` and the next state
        transition_matrix = Tensor(torch.randn(2, 2, requires_grad=True))
        self.trans = f_dist.MultivariateNormal(loc=Tensor(torch.randn(2)), scale_tril=Tensor(torch.eye(2)))

        # This is what we want
        # self.obs = matrix_and_mvn_to_funsor(observation_matrix, observation_dist,
        #                                     ("time",), "state(time=1)", "value")
        # but instead we will manually insert a bias term into obs.
        bias_matrix = torch.eye(num_sensors * 2)
        obs_and_bias_matrix = torch.cat((observation_matrix, bias_matrix))
        self.obs = f_dist.MultivariateNormal(loc=Tensor(torch.randn(2)), scale_tril=Tensor(torch.eye(2)))

        # HACK to replace state_and_bias with two variables.
#         assert (not obs.deltas and
#                 isinstance(obs.discrete, Tensor) and
#                 isinstance(obs.gaussian, Gaussian))
#         inputs = OrderedDict()
#         for k, d in obs.gaussian.inputs.items():
#           if k == "state_and_bias":
#             assert d == reals(2 + num_sensors * 2), d
#             inputs["state(time=1)"] = reals(2)
#             inputs["bias"] = reals(num_sensors * 2)
#           else:
#             inputs[k] = d
#         g = obs.gaussian
#         self.obs = obs.discrete + Gaussian(g.info_vec, g.precision, inputs)

        # we add bias to the observation as a global variable
#         data = torch.stack([frame["x"] for frame in track])
#         data = Tensor(data.unsqueeze(1)
#                           .expand(-1, self.num_sensors, 2)
#                           .reshape(-1, self.num_sensors * 2),
#                       OrderedDict(time=bint(len(track))))
        assert isinstance(track, Tensor)
        data = Tensor(track.data.reshape(-1, self.num_sensors * 2),
                      OrderedDict(time=bint(len(track.data))))

        with interpretation(eager_or_die):
            logp = self.obs(value=data)
            logp += bias_dist + self.trans

            # collapse out the time variable
            # TODO this can only handle homogeneous funsor types
            logp = sequential_sum_product(ops.logaddexp, ops.add,
                                          logp, "time", {"state": "state(time=1)"})
            logp += self.init
            # marginalize out remaining latent variables
            logp = logp.reduce(ops.logaddexp)

        # extract torch.Tensor from funsor
        assert isinstance(logp, Tensor), logp.pretty()
        return logp.data

def main(args):
    bias_scales = torch.ones(2, requires_grad=True)  # This can be learned
    params = [bias_scales]
    params_snapshot = []
    losses = []
    # params.append(transition_matrix)
    optim = Adam(params, lr=0.1)
    data = generate_data()
    model = HMM(num_sensors)
    for i in range(args.num_epochs):
        optim.zero_grad()
        log_prob = model(full_observations)
        loss = -log_prob
        loss.backward()
        losses.append(loss.item())
        if i % 10 == 0:
            params_snapshot.append(bias_scales.data.clone().cpu().numpy())
            print(loss.item())
        optim.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Switching linear dynamical system")
    parser.add_argument("-n", "--num-epochs", default=199, type=int)
    args = parser.parse_args()
    main(args)
