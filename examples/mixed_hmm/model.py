# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import pyro
import torch
from torch.distributions import constraints

import funsor.torch.distributions as dist
import funsor.ops as ops
from funsor.domains import Bint, Reals
from funsor.tensor import Tensor
from funsor.terms import Stack, Variable, to_funsor


class Guide(object):
    """generic mean-field guide for continuous random effects"""

    def __init__(self, config):
        self.config = config
        self.params = self.initialize_params()

    def initialize_params(self):

        # dictionary of guide random effect parameters
        params = {
            "eps_g": {},
            "eps_i": {},
        }

        N_state = self.config["sizes"]["state"]

        # initialize group-level parameters
        if self.config["group"]["random"] == "continuous":

            params["eps_g"]["loc"] = Tensor(
                pyro.param("loc_group", lambda: torch.zeros((N_state, N_state))),
                OrderedDict([("y_prev", Bint[N_state])]),
            )

            params["eps_g"]["scale"] = Tensor(
                pyro.param(
                    "scale_group",
                    lambda: torch.ones((N_state, N_state)),
                    constraint=constraints.positive,
                ),
                OrderedDict([("y_prev", Bint[N_state])]),
            )

        # initialize individual-level random effect parameters
        N_c = self.config["sizes"]["group"]
        if self.config["individual"]["random"] == "continuous":

            params["eps_i"]["loc"] = Tensor(
                pyro.param(
                    "loc_individual", lambda: torch.zeros((N_c, N_state, N_state))
                ),
                OrderedDict([("g", Bint[N_c]), ("y_prev", Bint[N_state])]),
            )

            params["eps_i"]["scale"] = Tensor(
                pyro.param(
                    "scale_individual",
                    lambda: torch.ones((N_c, N_state, N_state)),
                    constraint=constraints.positive,
                ),
                OrderedDict([("g", Bint[N_c]), ("y_prev", Bint[N_state])]),
            )

        self.params = params
        return self.params

    def __call__(self):

        # calls pyro.param so that params are exposed and constraints applied
        # should not create any new torch.Tensors after __init__
        self.initialize_params()

        N_c = self.config["sizes"]["group"]
        N_s = self.config["sizes"]["individual"]

        log_prob = Tensor(torch.tensor(0.0), OrderedDict())

        plate_g = Tensor(torch.zeros(N_c), OrderedDict([("g", Bint[N_c])]))
        plate_i = Tensor(torch.zeros(N_s), OrderedDict([("i", Bint[N_s])]))

        if self.config["group"]["random"] == "continuous":
            eps_g_dist = plate_g + dist.Normal(**self.params["eps_g"])(value="eps_g")
            log_prob += eps_g_dist

        # individual-level random effects
        if self.config["individual"]["random"] == "continuous":
            eps_i_dist = (
                plate_g + plate_i + dist.Normal(**self.params["eps_i"])(value="eps_i")
            )
            log_prob += eps_i_dist

        return log_prob


class Model(object):
    def __init__(self, config):
        self.config = config
        self.params = self.initialize_params()
        self.raggedness_masks = self.initialize_raggedness_masks()
        self.observations = self.initialize_observations()

    def initialize_params(self):

        # return a dict of per-site params as funsor.tensor.Tensors
        params = {
            "e_g": {},
            "theta_g": {},
            "eps_g": {},
            "e_i": {},
            "theta_i": {},
            "eps_i": {},
            "zi_step": {},
            "step": {},
            "angle": {},
            "zi_omega": {},
            "omega": {},
        }

        # size parameters
        N_v = self.config["sizes"]["random"]
        N_state = self.config["sizes"]["state"]

        # initialize group-level random effect parameters
        if self.config["group"]["random"] == "discrete":

            params["e_g"]["probs"] = Tensor(
                pyro.param(
                    "probs_e_g",
                    lambda: torch.randn((N_v,)).abs(),
                    constraint=constraints.simplex,
                ),
                OrderedDict(),
            )

            params["eps_g"]["theta"] = Tensor(
                pyro.param("theta_g", lambda: torch.randn((N_v, N_state, N_state))),
                OrderedDict([("e_g", Bint[N_v]), ("y_prev", Bint[N_state])]),
            )

        elif self.config["group"]["random"] == "continuous":

            # note these are prior values, trainable versions live in guide
            params["eps_g"]["loc"] = Tensor(
                torch.zeros((N_state, N_state)),
                OrderedDict([("y_prev", Bint[N_state])]),
            )

            params["eps_g"]["scale"] = Tensor(
                torch.ones((N_state, N_state)),
                OrderedDict([("y_prev", Bint[N_state])]),
            )

        # initialize individual-level random effect parameters
        N_c = self.config["sizes"]["group"]
        if self.config["individual"]["random"] == "discrete":

            params["e_i"]["probs"] = Tensor(
                pyro.param(
                    "probs_e_i",
                    lambda: torch.randn(
                        (
                            N_c,
                            N_v,
                        )
                    ).abs(),
                    constraint=constraints.simplex,
                ),
                OrderedDict([("g", Bint[N_c])]),  # different value per group
            )

            params["eps_i"]["theta"] = Tensor(
                pyro.param(
                    "theta_i", lambda: torch.randn((N_c, N_v, N_state, N_state))
                ),
                OrderedDict(
                    [("g", Bint[N_c]), ("e_i", Bint[N_v]), ("y_prev", Bint[N_state])]
                ),
            )

        elif self.config["individual"]["random"] == "continuous":

            params["eps_i"]["loc"] = Tensor(
                torch.zeros((N_c, N_state, N_state)),
                OrderedDict([("g", Bint[N_c]), ("y_prev", Bint[N_state])]),
            )

            params["eps_i"]["scale"] = Tensor(
                torch.ones((N_c, N_state, N_state)),
                OrderedDict([("g", Bint[N_c]), ("y_prev", Bint[N_state])]),
            )

        # initialize likelihood parameters
        # observation 1: step size (step ~ Gamma)
        params["zi_step"]["zi_param"] = Tensor(
            pyro.param(
                "step_zi_param",
                lambda: torch.ones((N_state, 2)),
                constraint=constraints.simplex,
            ),
            OrderedDict([("y_curr", Bint[N_state])]),
        )

        params["step"]["concentration"] = Tensor(
            pyro.param(
                "step_param_concentration",
                lambda: torch.randn((N_state,)).abs(),
                constraint=constraints.positive,
            ),
            OrderedDict([("y_curr", Bint[N_state])]),
        )

        params["step"]["rate"] = Tensor(
            pyro.param(
                "step_param_rate",
                lambda: torch.randn((N_state,)).abs(),
                constraint=constraints.positive,
            ),
            OrderedDict([("y_curr", Bint[N_state])]),
        )

        # observation 2: step angle (angle ~ VonMises)
        params["angle"]["concentration"] = Tensor(
            pyro.param(
                "angle_param_concentration",
                lambda: torch.randn((N_state,)).abs(),
                constraint=constraints.positive,
            ),
            OrderedDict([("y_curr", Bint[N_state])]),
        )

        params["angle"]["loc"] = Tensor(
            pyro.param("angle_param_loc", lambda: torch.randn((N_state,)).abs()),
            OrderedDict([("y_curr", Bint[N_state])]),
        )

        # observation 3: dive activity (omega ~ Beta)
        params["zi_omega"]["zi_param"] = Tensor(
            pyro.param(
                "omega_zi_param",
                lambda: torch.ones((N_state, 2)),
                constraint=constraints.simplex,
            ),
            OrderedDict([("y_curr", Bint[N_state])]),
        )

        params["omega"]["concentration0"] = Tensor(
            pyro.param(
                "omega_param_concentration0",
                lambda: torch.randn((N_state,)).abs(),
                constraint=constraints.positive,
            ),
            OrderedDict([("y_curr", Bint[N_state])]),
        )

        params["omega"]["concentration1"] = Tensor(
            pyro.param(
                "omega_param_concentration1",
                lambda: torch.randn((N_state,)).abs(),
                constraint=constraints.positive,
            ),
            OrderedDict([("y_curr", Bint[N_state])]),
        )

        self.params = params
        return self.params

    def initialize_observations(self):
        """
        Convert raw observation tensors into funsor.tensor.Tensors
        """
        batch_inputs = OrderedDict(
            [
                ("i", Bint[self.config["sizes"]["individual"]]),
                ("g", Bint[self.config["sizes"]["group"]]),
                ("t", Bint[self.config["sizes"]["timesteps"]]),
            ]
        )

        observations = {}
        for name, data in self.config["observations"].items():
            observations[name] = Tensor(
                data[..., : self.config["sizes"]["timesteps"]], batch_inputs
            )

        self.observations = observations
        return self.observations

    def initialize_raggedness_masks(self):
        """
        Convert raw raggedness tensors into funsor.tensor.Tensors
        """
        batch_inputs = OrderedDict(
            [
                ("i", Bint[self.config["sizes"]["individual"]]),
                ("g", Bint[self.config["sizes"]["group"]]),
                ("t", Bint[self.config["sizes"]["timesteps"]]),
            ]
        )

        raggedness_masks = {}
        for name in ("individual", "timestep"):
            data = self.config[name]["mask"]
            if len(data.shape) < len(batch_inputs):
                while len(data.shape) < len(batch_inputs):
                    data = data.unsqueeze(-1)
                data = data.expand(tuple(v.dtype for v in batch_inputs.values()))
            data = data.to(self.config["observations"]["step"].dtype)
            raggedness_masks[name] = Tensor(
                data[..., : self.config["sizes"]["timesteps"]], batch_inputs
            )

        self.raggedness_masks = raggedness_masks
        return self.raggedness_masks

    def __call__(self):

        # calls pyro.param so that params are exposed and constraints applied
        # should not create any new torch.Tensors after __init__
        self.initialize_params()

        N_state = self.config["sizes"]["state"]

        # initialize gamma to uniform
        gamma = Tensor(
            torch.zeros((N_state, N_state)),
            OrderedDict([("y_prev", Bint[N_state])]),
        )

        N_v = self.config["sizes"]["random"]
        N_c = self.config["sizes"]["group"]
        log_prob = []

        plate_g = Tensor(torch.zeros(N_c), OrderedDict([("g", Bint[N_c])]))

        # group-level random effects
        if self.config["group"]["random"] == "discrete":
            # group-level discrete effect
            e_g = Variable("e_g", Bint[N_v])
            e_g_dist = plate_g + dist.Categorical(**self.params["e_g"])(value=e_g)

            log_prob.append(e_g_dist)

            eps_g = (plate_g + self.params["eps_g"]["theta"])(e_g=e_g)

        elif self.config["group"]["random"] == "continuous":
            eps_g = Variable("eps_g", Reals[N_state])
            eps_g_dist = plate_g + dist.Normal(**self.params["eps_g"])(value=eps_g)

            log_prob.append(eps_g_dist)
        else:
            eps_g = to_funsor(0.0)

        N_s = self.config["sizes"]["individual"]

        plate_i = Tensor(torch.zeros(N_s), OrderedDict([("i", Bint[N_s])]))
        # individual-level random effects
        if self.config["individual"]["random"] == "discrete":
            # individual-level discrete effect
            e_i = Variable("e_i", Bint[N_v])
            e_i_dist = (
                plate_g
                + plate_i
                + dist.Categorical(**self.params["e_i"])(value=e_i)
                * self.raggedness_masks["individual"](t=0)
            )

            log_prob.append(e_i_dist)

            eps_i = plate_i + plate_g + self.params["eps_i"]["theta"](e_i=e_i)

        elif self.config["individual"]["random"] == "continuous":
            eps_i = Variable("eps_i", Reals[N_state])
            eps_i_dist = (
                plate_g + plate_i + dist.Normal(**self.params["eps_i"])(value=eps_i)
            )

            log_prob.append(eps_i_dist)
        else:
            eps_i = to_funsor(0.0)

        # add group-level and individual-level random effects to gamma
        gamma = gamma + eps_g + eps_i

        N_state = self.config["sizes"]["state"]

        # we've accounted for all effects, now actually compute gamma_y
        gamma_y = gamma(y_prev="y(t=1)")

        y = Variable("y", Bint[N_state])
        y_dist = (
            plate_g
            + plate_i
            + dist.Categorical(probs=gamma_y.exp() / gamma_y.exp().sum())(value=y)
        )

        # observation 1: step size
        step_dist = (
            plate_g
            + plate_i
            + dist.Gamma(**{k: v(y_curr=y) for k, v in self.params["step"].items()})(
                value=self.observations["step"]
            )
        )

        # step size zero-inflation
        if self.config["zeroinflation"]:
            step_zi = dist.Categorical(
                probs=self.params["zi_step"]["zi_param"](y_curr=y)
            )(value="zi_step")
            step_zi_dist = (
                plate_g
                + plate_i
                + dist.Delta(self.config["MISSING"], 0.0)(
                    value=self.observations["step"]
                )
            )
            step_dist = (step_zi + Stack("zi_step", (step_dist, step_zi_dist))).reduce(
                ops.logaddexp, "zi_step"
            )

        # observation 2: step angle
        angle_dist = (
            plate_g
            + plate_i
            + dist.VonMises(
                **{k: v(y_curr=y) for k, v in self.params["angle"].items()}
            )(value=self.observations["angle"])
        )

        # observation 3: dive activity
        omega_dist = (
            plate_g
            + plate_i
            + dist.Beta(**{k: v(y_curr=y) for k, v in self.params["omega"].items()})(
                value=self.observations["omega"]
            )
        )

        # dive activity zero-inflation
        if self.config["zeroinflation"]:
            omega_zi = dist.Categorical(
                probs=self.params["zi_omega"]["zi_param"](y_curr=y)
            )(value="zi_omega")
            omega_zi_dist = (
                plate_g
                + plate_i
                + dist.Delta(self.config["MISSING"], 0.0)(
                    value=self.observations["omega"]
                )
            )
            omega_dist = (
                omega_zi + Stack("zi_omega", (omega_dist, omega_zi_dist))
            ).reduce(ops.logaddexp, "zi_omega")

        # finally, construct the term for parallel scan reduction
        hmm_factor = step_dist + angle_dist + omega_dist
        hmm_factor = hmm_factor * self.raggedness_masks["individual"]
        hmm_factor = hmm_factor * self.raggedness_masks["timestep"]
        # copy masking behavior of pyro.infer.TraceEnum_ELBO._compute_model_factors
        hmm_factor = hmm_factor + y_dist
        log_prob.insert(0, hmm_factor)

        return log_prob
