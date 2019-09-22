from collections import OrderedDict

import torch
from torch.distributions import constraints

import pyro

import funsor.distributions as dist
from funsor.domains import bint, reals
from funsor.terms import Number, to_funsor
from funsor.torch import Tensor


def initialize_guide_params(config):

    # dictionary of guide random effect parameters
    params = {
        "eps_g": {},
        "eps_i": {},
    }

    N_state = config["sizes"]["state"]

    # initialize group-level parameters
    if config["group"]["random"] == "continuous":

        params["eps_g"]["loc"] = Tensor(
            pyro.param("loc_group",
                       lambda: torch.zeros((N_state, N_state))),
            OrderedDict([("y_prev", bint(N_state))]),
            reals(N_state)
        )

        params["eps_g"]["scale"] = Tensor(
            pyro.param("scale_group",
                       lambda: torch.ones((N_state, N_state)),
                       constraint=constraints.positive),
            OrderedDict([("y_prev", bint(N_state))]),
            reals(N_state)
        )

    # initialize individual-level random effect parameters
    N_c = config["sizes"]["group"]
    if config["individual"]["random"] == "continuous":

        params["eps_i"]["loc"] = Tensor(
            pyro.param("loc_individual",
                       lambda: torch.zeros((N_c, N_state, N_state))),
            OrderedDict([("g", bint(N_c)), ("y_prev", bint(N_state))]),
            reals(N_state)
        )

        params["eps_i"]["scale"] = Tensor(
            pyro.param("scale_individual",
                       lambda: torch.ones((N_c, N_state, N_state)),
                       constraint=constraints.positive),
            OrderedDict([("g", bint(N_c)), ("y_prev", bint(N_state))]),
            reals(N_state)
        )

    return params


def initialize_model_params(config):

    # return a dict of per-site params as funsor.torch.Tensors
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
    N_v = config["sizes"]["random"]
    N_state = config["sizes"]["state"]

    # initialize group-level random effect parameters
    if config["group"]["random"] == "discrete":

        params["e_g"]["probs"] = Tensor(
            pyro.param("probs_e_g",
                       lambda: torch.randn((N_v,)).abs(),
                       constraint=constraints.simplex),
            OrderedDict(),
            reals(N_v)
        )

        params["eps_g"]["theta"] = Tensor(
            pyro.param("theta_g",
                       lambda: torch.randn((N_v, N_state, N_state))),
            OrderedDict([("e_g", bint(N_v)), ("y_prev", bint(N_state))]),
            reals(N_state)
        )

    elif config["group"]["random"] == "continuous":

        # note these are prior values, trainable versions live in guide
        params["eps_g"]["loc"] = Tensor(
            torch.zeros((N_state, N_state)),
            OrderedDict([("y_prev", bint(N_state))]),
            reals(N_state)
        )

        params["eps_g"]["scale"] = Tensor(
            torch.ones((N_state, N_state)),
            OrderedDict([("y_prev", bint(N_state))]),
            reals(N_state)
        )

    # initialize individual-level random effect parameters
    N_c = config["sizes"]["group"]
    if config["individual"]["random"] == "discrete":

        params["e_i"]["probs"] = Tensor(
            pyro.param("probs_e_i",
                       lambda: torch.randn((N_c, N_v,)).abs(),
                       constraint=constraints.simplex),
            OrderedDict([("g", bint(N_c))]),  # different value per group
            reals(N_v)
        )

        params["eps_i"]["theta"] = Tensor(
            pyro.param("theta_i",
                       lambda: torch.randn((N_c, N_v, N_state, N_state))),
            OrderedDict([("g", bint(N_c), "e_i", bint(N_v)), ("y_prev", bint(N_state))]),
            reals(N_state)
        )

    elif config["individual"]["random"] == "continuous":

        params["eps_i"]["loc"] = Tensor(
            torch.zeros((N_c, N_state, N_state)),
            OrderedDict([("g", bint(N_c)), ("y_prev", bint(N_state))]),
            reals(N_state)
        )

        params["eps_i"]["scale"] = Tensor(
            torch.ones((N_c, N_state, N_state)),
            OrderedDict([("g", bint(N_c)), ("y_prev", bint(N_state))]),
            reals(N_state)
        )

    # initialize likelihood parameters
    # observation 1: step size (step ~ Gamma)
    params["zi_step"]["zi_param"] = Tensor(
        pyro.param("step_zi_param",
                   lambda: torch.ones((N_state, 2))),
        OrderedDict([("y_curr", bint(N_state))]),
        reals(2),
    )

    params["step"]["concentration"] = Tensor(
        pyro.param("step_param_concentration",
                   lambda: torch.randn((N_state,)).abs(),
                   constraint=constraints.positive),
        OrderedDict([("y_curr", bint(N_state))]),
        reals()
    )

    params["step"]["rate"] = Tensor(
        pyro.param("step_param_rate",
                   lambda: torch.randn((N_state,)).abs(),
                   constraint=constraints.positive),
        OrderedDict([("y_curr", bint(N_state))]),
        reals()
    )

    # observation 2: step angle (angle ~ VonMises)
    params["angle"]["concentration"] = Tensor(
        pyro.param("angle_param_concentration",
                   lambda: torch.randn((N_state,)).abs(),
                   constraint=constraints.positive),
        OrderedDict([("y_curr", bint(N_state))]),
        reals()
    )

    params["angle"]["loc"] = Tensor(
        pyro.param("angle_param_loc",
                   lambda: torch.randn((N_state,)).abs()),
        OrderedDict([("y_curr", bint(N_state))]),
        reals()
    )

    # observation 3: dive activity (omega ~ Beta)
    params["zi_omega"]["zi_param"] = Tensor(
        pyro.param("omega_zi_param",
                   lambda: torch.ones((N_state, 2))),
        OrderedDict([("y_curr", bint(N_state))]),
        reals(2)
    )

    params["omega"]["concentration0"] = Tensor(
        pyro.param("omega_param_concentration0",
                   lambda: torch.randn((N_state,)).abs(),
                   constraint=constraints.positive),
        OrderedDict([("y_curr", bint(N_state))]),
        reals()
    )

    params["omega"]["cencentration1"] = Tensor(
        pyro.param("omega_param_concentration1",
                   lambda: torch.randn((N_state,)).abs(),
                   constraint=constraints.positive),
        OrderedDict([("y_curr", bint(N_state))]),
        reals()
    )

    return params


def initialize_observations(config):
    """
    Convert raw observation tensors into funsor.torch.Tensors
    """
    batch_inputs = OrderedDict([
        ("i", config["sizes"]["individual"]),
        ("g", config["sizes"]["group"]),
        ("t", config["sizes"]["timesteps"]),
    ])

    observations = {}
    for name, data in config["observations"].items():
        observations[name] = Tensor(data, batch_inputs)

    return observations


def guide_simple(config):
    """generic mean-field guide for continuous random effects"""
    params = initialize_guide_params(config)

    N_c = config["sizes"]["group"]
    with pyro.plate("group", N_c, dim=-1):

        if config["group"]["random"] == "continuous":
            pyro.sample("eps_g", dist.Normal(**params["eps_g"]).to_event(1),
                        )  # infer={"num_samples": 10})

        N_s = config["sizes"]["individual"]
        with pyro.plate("individual", N_s, dim=-2):  # , poutine.mask(mask=config["individual"]["mask"]):

            # individual-level random effects
            if config["individual"]["random"] == "continuous":
                pyro.sample("eps_i", dist.Normal(**params["eps_i"]).to_event(1),
                            )  # infer={"num_samples": 10})


def model_simple(config):
    """
    Simpler version of generic model with no zero-inflation
    """

    # MISSING = config["MISSING"]  # used for masking and zero-inflation
    N_state = config["sizes"]["state"]

    params = initialize_model_params(config)
    observations = initialize_observations(config)

    # initialize gamma to uniform
    gamma = Tensor(
        torch.zeros((N_state, N_state)),
        OrderedDict([("y_prev", bint(N_state))]),  # , ("y_curr", bint(N_state))]),
        reals(N_state),  # reals()
    )

    N_c = config["sizes"]["group"]
    with pyro.plate("group", N_c, dim=-1):

        # group-level random effects
        if config["group"]["random"] == "discrete":
            # group-level discrete effect
            e_g = pyro.sample("e_g", dist.Categorical(**params["e_g"]))

            # TODO use positional index or use non-y name
            eps_g = params["theta_g"]["theta"](e_g=e_g)

        elif config["group"]["random"] == "continuous":
            # TODO use Independent() here and shape like matrix
            eps_g = pyro.sample("eps_g", dist.Normal(**params["eps_g"]).to_event(1),
                                )  # infer={"num_samples": 10})
        else:
            eps_g = to_funsor(0.)

        # add group-level random effect to gamma
        gamma = gamma + eps_g

        N_s = config["sizes"]["individual"]

        # TODO replace mask with site-specific masks via .mask()
        with pyro.plate("individual", N_s, dim=-2):  # , poutine.mask(mask=config["individual"]["mask"]):

            # individual-level random effects
            if config["individual"]["random"] == "discrete":
                # individual-level discrete effect
                e_i = pyro.sample("e_i", dist.Categorical(**params["e_i"]))

                eps_i = params["theta_i"]["theta"](e_i=e_i)

            elif config["individual"]["random"] == "continuous":
                # TODO use Independent() here and shape like matrix
                eps_i = pyro.sample("eps_i", dist.Normal(**params["eps_i"]).to_event(1),
                                    )  # infer={"num_samples": 10})
            else:
                eps_i = to_funsor(0.)

            # add individual-level random effect to gamma
            gamma = gamma + eps_i

            # y = torch.tensor(0).long()
            y = Number(0, bint(config["sizes"]["state"]))

            N_t = config["sizes"]["timesteps"]
            for t in range(N_t):  # pyro.markov(range(N_t)):
                # TODO replace with site-specific masks via .mask()
                # with poutine.mask(mask=config["timestep"]["mask"][..., t]):

                gamma_t = gamma  # per-timestep variable

                # we've accounted for all effects, now actually compute gamma_y
                gamma_y = gamma_t(y_prev=y)
                y = pyro.sample("y_{}".format(t), dist.Categorical(logits=gamma_y))

                # observation 1: step size
                step_dist = dist.Gamma(
                    **{k: v(y_curr=y) for k, v in params["step"].items()}
                )
                pyro.sample("step_{}".format(t),
                            step_dist,
                            obs=observations["step"][..., t])

                # observation 2: step angle
                angle_dist = dist.VonMises(
                    **{k: v(y_curr=y) for k, v in params["angle"].items()}
                )
                pyro.sample("angle_{}".format(t),
                            angle_dist,
                            obs=observations["angle"][..., t])

                # observation 3: dive activity
                omega_dist = dist.Beta(
                    **{k: v(y_curr=y) for k, v in params["omega"].items()}
                )
                pyro.sample("omega_{}".format(t),
                            omega_dist,
                            obs=observations["omega"][..., t])
