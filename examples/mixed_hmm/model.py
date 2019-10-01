from collections import OrderedDict

import torch
from torch.distributions import constraints

import pyro

import funsor.distributions as dist
from funsor.domains import bint, reals
from funsor.interpreter import interpretation
from funsor.terms import Number, Variable, eager, to_funsor
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
        )

        params["eps_g"]["scale"] = Tensor(
            pyro.param("scale_group",
                       lambda: torch.ones((N_state, N_state)),
                       constraint=constraints.positive),
            OrderedDict([("y_prev", bint(N_state))]),
        )

    # initialize individual-level random effect parameters
    N_c = config["sizes"]["group"]
    if config["individual"]["random"] == "continuous":

        params["eps_i"]["loc"] = Tensor(
            pyro.param("loc_individual",
                       lambda: torch.zeros((N_c, N_state, N_state))),
            OrderedDict([("g", bint(N_c)), ("y_prev", bint(N_state))]),
        )

        params["eps_i"]["scale"] = Tensor(
            pyro.param("scale_individual",
                       lambda: torch.ones((N_c, N_state, N_state)),
                       constraint=constraints.positive),
            OrderedDict([("g", bint(N_c)), ("y_prev", bint(N_state))]),
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
        )

        params["eps_g"]["theta"] = Tensor(
            pyro.param("theta_g",
                       lambda: torch.randn((N_v, N_state, N_state))),
            OrderedDict([("e_g", bint(N_v)), ("y_prev", bint(N_state))]),
        )

    elif config["group"]["random"] == "continuous":

        # note these are prior values, trainable versions live in guide
        params["eps_g"]["loc"] = Tensor(
            torch.zeros((N_state, N_state)),
            OrderedDict([("y_prev", bint(N_state))]),
        )

        params["eps_g"]["scale"] = Tensor(
            torch.ones((N_state, N_state)),
            OrderedDict([("y_prev", bint(N_state))]),
        )

    # initialize individual-level random effect parameters
    N_c = config["sizes"]["group"]
    if config["individual"]["random"] == "discrete":

        params["e_i"]["probs"] = Tensor(
            pyro.param("probs_e_i",
                       lambda: torch.randn((N_c, N_v,)).abs(),
                       constraint=constraints.simplex),
            OrderedDict([("g", bint(N_c))]),  # different value per group
        )

        params["eps_i"]["theta"] = Tensor(
            pyro.param("theta_i",
                       lambda: torch.randn((N_c, N_v, N_state, N_state))),
            OrderedDict([("g", bint(N_c)), ("e_i", bint(N_v)), ("y_prev", bint(N_state))]),
        )

    elif config["individual"]["random"] == "continuous":

        params["eps_i"]["loc"] = Tensor(
            torch.zeros((N_c, N_state, N_state)),
            OrderedDict([("g", bint(N_c)), ("y_prev", bint(N_state))]),
        )

        params["eps_i"]["scale"] = Tensor(
            torch.ones((N_c, N_state, N_state)),
            OrderedDict([("g", bint(N_c)), ("y_prev", bint(N_state))]),
        )

    # initialize likelihood parameters
    # observation 1: step size (step ~ Gamma)
    params["zi_step"]["zi_param"] = Tensor(
        pyro.param("step_zi_param",
                   lambda: torch.ones((N_state, 2)),
                   constraint=constraints.simplex),
        OrderedDict([("y_curr", bint(N_state))]),
    )

    params["step"]["concentration"] = Tensor(
        pyro.param("step_param_concentration",
                   lambda: torch.randn((N_state,)).abs(),
                   constraint=constraints.positive),
        OrderedDict([("y_curr", bint(N_state))]),
    )

    params["step"]["rate"] = Tensor(
        pyro.param("step_param_rate",
                   lambda: torch.randn((N_state,)).abs(),
                   constraint=constraints.positive),
        OrderedDict([("y_curr", bint(N_state))]),
    )

    # observation 2: step angle (angle ~ VonMises)
    params["angle"]["concentration"] = Tensor(
        pyro.param("angle_param_concentration",
                   lambda: torch.randn((N_state,)).abs(),
                   constraint=constraints.positive),
        OrderedDict([("y_curr", bint(N_state))]),
    )

    params["angle"]["loc"] = Tensor(
        pyro.param("angle_param_loc",
                   lambda: torch.randn((N_state,)).abs()),
        OrderedDict([("y_curr", bint(N_state))]),
    )

    # observation 3: dive activity (omega ~ Beta)
    params["zi_omega"]["zi_param"] = Tensor(
        pyro.param("omega_zi_param",
                   lambda: torch.ones((N_state, 2)),
                   constraint=constraints.simplex),
        OrderedDict([("y_curr", bint(N_state))]),
    )

    params["omega"]["concentration0"] = Tensor(
        pyro.param("omega_param_concentration0",
                   lambda: torch.randn((N_state,)).abs(),
                   constraint=constraints.positive),
        OrderedDict([("y_curr", bint(N_state))]),
    )

    params["omega"]["concentration1"] = Tensor(
        pyro.param("omega_param_concentration1",
                   lambda: torch.randn((N_state,)).abs(),
                   constraint=constraints.positive),
        OrderedDict([("y_curr", bint(N_state))]),
    )

    return params


def initialize_observations(config):
    """
    Convert raw observation tensors into funsor.torch.Tensors
    """
    batch_inputs = OrderedDict([
        ("i", bint(config["sizes"]["individual"])),
        ("g", bint(config["sizes"]["group"])),
        ("t", bint(config["sizes"]["timesteps"])),
    ])

    observations = {}
    for name, data in config["observations"].items():
        observations[name] = Tensor(data[..., :config["sizes"]["timesteps"]], batch_inputs)

    return observations


def initialize_zi_masks(config):
    """
    Convert raw zero-inflation masks into funsor.torch.Tensors
    """
    batch_inputs = OrderedDict([
        ("i", bint(config["sizes"]["individual"])),
        ("g", bint(config["sizes"]["group"])),
        ("t", bint(config["sizes"]["timesteps"])),
    ])

    zi_masks = {}
    for name, data in config["observations"].items():
        mask = (data[..., :config["sizes"]["timesteps"]] == config["MISSING"]).to(
            data.dtype)
        zi_masks[name] = Tensor(1. - mask, batch_inputs)  # , bint(2))

    return zi_masks


def initialize_raggedness_masks(config):
    """
    Convert raw raggedness tensors into funsor.torch.Tensors
    """
    batch_inputs = OrderedDict([
        ("i", bint(config["sizes"]["individual"])),
        ("g", bint(config["sizes"]["group"])),
        ("t", bint(config["sizes"]["timesteps"])),
    ])

    raggedness_masks = {}
    for name in ("individual", "timestep"):
        data = config[name]["mask"]
        if len(data.shape) < len(batch_inputs):
            while len(data.shape) < len(batch_inputs):
                data = data.unsqueeze(-1)
            data = data.expand(tuple(v.dtype for v in batch_inputs.values()))
        data = 1. - data.to(config["observations"]["step"].dtype)
        raggedness_masks[name] = Tensor(data[..., :config["sizes"]["timesteps"]],
                                        batch_inputs)

    return raggedness_masks


def guide_sequential(config):
    """generic mean-field guide for continuous random effects"""
    params = initialize_guide_params(config)

    N_c = config["sizes"]["group"]
    N_s = config["sizes"]["individual"]

    log_prob = Tensor(torch.tensor(0.), OrderedDict())

    plate_g = Tensor(torch.zeros(N_c), OrderedDict([("g", bint(N_c))]))
    plate_i = Tensor(torch.zeros(N_s), OrderedDict([("i", bint(N_s))]))

    if config["group"]["random"] == "continuous":
        with interpretation(eager):
            eps_g_dist = plate_g + dist.Normal(**params["eps_g"])(value="eps_g")

        log_prob += eps_g_dist

    # with poutine.mask(mask=config["individual"]["mask"]):

    # individual-level random effects
    if config["individual"]["random"] == "continuous":
        with interpretation(eager):
            eps_i_dist = plate_g + plate_i + dist.Normal(**params["eps_i"])(value="eps_i")

        log_prob += eps_i_dist

    return log_prob


def model_sequential(config):
    """
    Simpler version of generic model with no zero-inflation
    """

    # MISSING = config["MISSING"]  # used for masking and zero-inflation
    N_state = config["sizes"]["state"]

    params = initialize_model_params(config)
    observations = initialize_observations(config)
    raggedness_masks = initialize_raggedness_masks(config)
    zi_masks = initialize_zi_masks(config)

    # initialize gamma to uniform
    gamma = Tensor(
        torch.zeros((N_state, N_state)),
        OrderedDict([("y_prev", bint(N_state))]),
    )

    N_v = config["sizes"]["random"]
    N_c = config["sizes"]["group"]
    log_prob = []  # Tensor(torch.tensor(0.), OrderedDict())

    # with pyro.plate("group", N_c, dim=-1):
    plate_g = Tensor(torch.zeros(N_c), OrderedDict([("g", bint(N_c))]))

    # group-level random effects
    if config["group"]["random"] == "discrete":
        # group-level discrete effect
        e_g = Variable("e_g", bint(N_v))
        with interpretation(eager):
            e_g_dist = plate_g + dist.Categorical(**params["e_g"])(value=e_g)

        log_prob.append(e_g_dist)

        eps_g = params["eps_g"]["theta"](e_g=e_g)

    elif config["group"]["random"] == "continuous":
        eps_g = Variable("eps_g", reals(N_state))
        with interpretation(eager):
            eps_g_dist = plate_g + dist.Normal(**params["eps_g"])(value=eps_g)

        log_prob.append(eps_g_dist)
    else:
        eps_g = to_funsor(0.)

    N_s = config["sizes"]["individual"]

    # TODO replace mask with site-specific masks via .mask()
    # with pyro.plate("individual", N_s, dim=-2):  # , poutine.mask(mask=config["individual"]["mask"]):
    plate_i = Tensor(torch.zeros(N_s), OrderedDict([("i", bint(N_s))]))
    # individual-level random effects
    if config["individual"]["random"] == "discrete":
        # individual-level discrete effect
        e_i = Variable("e_i", bint(N_v))
        with interpretation(eager):
            e_i_dist = plate_g + plate_i + dist.Categorical(
                **params["e_i"]
            )(value=e_i) * raggedness_masks["individual"](t=0)

        log_prob.append(e_i_dist)

        eps_i = (plate_i + plate_g + params["eps_i"]["theta"](e_i=e_i))

    elif config["individual"]["random"] == "continuous":
        eps_i = Variable("eps_i", reals(N_state))
        with interpretation(eager):
            eps_i_dist = plate_g + plate_i + dist.Normal(**params["eps_i"])(value=eps_i)

        log_prob.append(eps_i_dist)
    else:
        eps_i = to_funsor(0.)

    # add group-level and individual-level random effects to gamma
    # XXX should the terms get materialize()-d?
    with interpretation(eager):
        gamma = gamma + eps_g + eps_i

    # initialize y in a single state for now
    y = Number(0, config["sizes"]["state"])

    N_t = config["sizes"]["timesteps"]
    N_state = config["sizes"]["state"]
    for t in range(N_t):  # pyro.markov(range(N_t)):
        # TODO replace with site-specific masks via .mask()
        # with poutine.mask(mask=config["timestep"]["mask"][..., t]):

        gamma_t = gamma  # per-timestep variable

        # we've accounted for all effects, now actually compute gamma_y
        gamma_y = gamma_t(y_prev=y)

        y = Variable("y_{}".format(t), bint(N_state))
        with interpretation(eager):
            y_dist = plate_g + plate_i + dist.Categorical(
                probs=gamma_y.exp() / gamma_y.exp().sum()
            )(value=y)

        log_prob.append(y_dist)

        # observation 1: step size
        with interpretation(eager):
            step_zi = dist.Categorical(probs=params["zi_step"]["zi_param"](y_curr=y))(
                value=Tensor(zi_masks["step"].data, zi_masks["step"].inputs, 2)(t=t))
            step_dist = plate_g + plate_i + step_zi + dist.Gamma(
                **{k: v(y_curr=y) for k, v in params["step"].items()}
            )(value=observations["step"](t=t)) * zi_masks["step"](t=t)

        log_prob.append(step_dist)

        # observation 2: step angle
        with interpretation(eager):
            angle_dist = plate_g + plate_i + dist.VonMises(
                **{k: v(y_curr=y) for k, v in params["angle"].items()}
            )(value=observations["angle"](t=t))

        log_prob.append(angle_dist)

        # observation 3: dive activity
        with interpretation(eager):
            omega_zi = dist.Categorical(probs=params["zi_omega"]["zi_param"](y_curr=y))(
                value=Tensor(zi_masks["omega"].data, zi_masks["omega"].inputs, 2)(t=t))
            omega_dist = plate_g + plate_i + omega_zi + dist.Beta(
                **{k: v(y_curr=y) for k, v in params["omega"].items()}
            )(value=observations["omega"](t=t)) * zi_masks["omega"](t=t)

        log_prob.append(omega_dist)

    return log_prob


def model_parallel(config):
    """
    Simpler version of generic model with no zero-inflation
    """

    # MISSING = config["MISSING"]  # used for masking and zero-inflation
    N_state = config["sizes"]["state"]

    params = initialize_model_params(config)
    observations = initialize_observations(config)
    raggedness_masks = initialize_raggedness_masks(config)
    zi_masks = initialize_zi_masks(config)

    # initialize gamma to uniform
    gamma = Tensor(
        torch.zeros((N_state, N_state)),
        OrderedDict([("y_prev", bint(N_state))]),
    )

    N_v = config["sizes"]["random"]
    N_c = config["sizes"]["group"]
    log_prob = []  # Tensor(torch.tensor(0.), OrderedDict())

    # with pyro.plate("group", N_c, dim=-1):
    plate_g = Tensor(torch.zeros(N_c), OrderedDict([("g", bint(N_c))]))

    # group-level random effects
    if config["group"]["random"] == "discrete":
        # group-level discrete effect
        e_g = Variable("e_g", bint(N_v))
        with interpretation(eager):
            e_g_dist = plate_g + dist.Categorical(**params["e_g"])(value=e_g)

        log_prob.append(e_g_dist)

        eps_g = params["eps_g"]["theta"](e_g=e_g)

    elif config["group"]["random"] == "continuous":
        eps_g = Variable("eps_g", reals(N_state))
        with interpretation(eager):
            eps_g_dist = plate_g + dist.Normal(**params["eps_g"])(value=eps_g)

        log_prob.append(eps_g_dist)
    else:
        eps_g = to_funsor(0.)

    N_s = config["sizes"]["individual"]

    # TODO replace mask with site-specific masks via .mask()
    # with pyro.plate("individual", N_s, dim=-2):  # , poutine.mask(mask=config["individual"]["mask"]):
    plate_i = Tensor(torch.zeros(N_s), OrderedDict([("i", bint(N_s))]))
    # individual-level random effects
    if config["individual"]["random"] == "discrete":
        # individual-level discrete effect
        e_i = Variable("e_i", bint(N_v))
        with interpretation(eager):
            e_i_dist = plate_g + plate_i + dist.Categorical(
                **params["e_i"]
            )(value=e_i) * raggedness_masks["individual"](t=0)

        log_prob.append(e_i_dist)

        eps_i = (plate_i + plate_g + params["eps_i"]["theta"](e_i=e_i))

    elif config["individual"]["random"] == "continuous":
        eps_i = Variable("eps_i", reals(N_state))
        with interpretation(eager):
            eps_i_dist = plate_g + plate_i + dist.Normal(**params["eps_i"])(value=eps_i)

        log_prob.append(eps_i_dist)
    else:
        eps_i = to_funsor(0.)

    # add group-level and individual-level random effects to gamma
    # XXX should the terms get materialize()-d?
    with interpretation(eager):
        gamma = gamma + eps_g + eps_i

    # initialize y in a single state for now
    # y = Number(0, config["sizes"]["state"])

    N_state = config["sizes"]["state"]
    # TODO replace with site-specific masks via .mask()
    # with poutine.mask(mask=config["timestep"]["mask"][..., t]):

    # we've accounted for all effects, now actually compute gamma_y
    gamma_y = gamma(y_prev="y(t=1)")

    y = Variable("y", bint(N_state))
    with interpretation(eager):
        y_dist = plate_g + plate_i + dist.Categorical(
            probs=gamma_y.exp() / gamma_y.exp().sum()
        )(value=y)

    # observation 1: step size
    with interpretation(eager):
        step_zi = dist.Categorical(probs=params["zi_step"]["zi_param"](y_curr=y))(
            value=Tensor(zi_masks["step"].data, zi_masks["step"].inputs, 2))
        step_dist = plate_g + plate_i + step_zi + dist.Gamma(
            **{k: v(y_curr=y) for k, v in params["step"].items()}
        )(value=observations["step"]) * zi_masks["step"]

    # observation 2: step angle
    with interpretation(eager):
        angle_dist = plate_g + plate_i + dist.VonMises(
            **{k: v(y_curr=y) for k, v in params["angle"].items()}
        )(value=observations["angle"])

    # observation 3: dive activity
    with interpretation(eager):
        omega_zi = dist.Categorical(probs=params["zi_omega"]["zi_param"](y_curr=y))(
            value=Tensor(zi_masks["omega"].data, zi_masks["omega"].inputs, 2))
        omega_dist = plate_g + plate_i + omega_zi + dist.Beta(
            **{k: v(y_curr=y) for k, v in params["omega"].items()}
        )(value=observations["omega"]) * zi_masks["omega"]

    with interpretation(eager):
        # construct the term for parallel scan reduction
        hmm_factor = y_dist + step_dist + angle_dist + omega_dist
        # hmm_factor = hmm_factor * raggedness_masks["individual"]
        # hmm_factor = hmm_factor * raggedness_masks["timestep"]
        log_prob.insert(0, hmm_factor)

    return log_prob
