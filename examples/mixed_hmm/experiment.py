# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import functools
import json
import os
import uuid

import pyro
import pyro.poutine as poutine
import torch

import funsor
import funsor.ops as ops
from funsor.interpreter import interpretation
from funsor.optimizer import apply_optimizer
from funsor.sum_product import MarkovProduct, naive_sequential_sum_product, sum_product
from funsor.terms import lazy, to_funsor
from model import Guide, Model
from seal_data import prepare_fake, prepare_seal


def aic_num_parameters(model, guide=None):
    """
    hacky AIC param count that includes all parameters in the model and guide
    """
    with poutine.block(), poutine.trace(param_only=True) as param_capture:
        model()
        if guide is not None:
            guide()

    return sum(node["value"].numel() for node in param_capture.trace.nodes.values())


def parallel_loss_fn(model, guide, parallel=True):
    # We're doing exact inference, so we don't use the guide here.
    factors = model()
    t_term, new_factors = factors[0], factors[1:]
    t = to_funsor("t", t_term.inputs["t"])
    if parallel:
        result = MarkovProduct(ops.logaddexp, ops.add, t_term, t, {"y(t=1)": "y"})
    else:
        result = naive_sequential_sum_product(
            ops.logaddexp, ops.add, t_term, t, {"y(t=1)": "y"}
        )
    new_factors = [result] + new_factors

    plates = frozenset(["g", "i"])
    eliminate = frozenset().union(*(f.inputs for f in new_factors))
    with interpretation(lazy):
        loss = sum_product(ops.logaddexp, ops.add, new_factors, eliminate, plates)
    loss = apply_optimizer(loss)
    assert not loss.inputs
    return -loss.data


def run_expt(args):
    funsor.set_backend("torch")

    optim = args["optim"]
    lr = args["learnrate"]
    schedule = (
        [] if not args["schedule"] else [int(i) for i in args["schedule"].split(",")]
    )
    # default these to "none" instead of None, which argparse does for some reason
    args["group"] = "none" if args["group"] is None else args["group"]
    args["individual"] = "none" if args["individual"] is None else args["individual"]
    random_effects = {"group": args["group"], "individual": args["individual"]}

    pyro.enable_validation(args["validation"])
    pyro.set_rng_seed(args["seed"])  # reproducible random effect parameter init
    if args["cuda"]:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    if args["dataset"] == "seal":
        filename = os.path.join(args["folder"], "prep_seal_data.csv")
        config = prepare_seal(filename, random_effects)
    elif args["dataset"] == "fake":
        fake_sizes = {
            "state": args["size_state"],
            "random": args["size_random"],
            "group": args["size_group"],
            "individual": args["size_individual"],
            "timesteps": args["size_timesteps"],
        }
        config = prepare_fake(fake_sizes, random_effects)
    else:
        raise ValueError("Dataset {} not yet included".format(args["dataset"]))

    if args["smoke"]:
        args["timesteps"] = 2
        config["sizes"]["timesteps"] = 3

    if args["truncate"] > 0:
        config["sizes"]["timesteps"] = args["truncate"]

    config["zeroinflation"] = args["zeroinflation"]

    model = Model(config)
    guide = Guide(config)
    loss_fn = parallel_loss_fn

    if args["jit"]:
        loss_fn = torch.jit.trace(lambda: loss_fn(model, guide, args["parallel"]), ())
    else:
        loss_fn = functools.partial(loss_fn, model, guide, args["parallel"])

    # count the number of parameters once
    num_parameters = aic_num_parameters(model, guide)

    losses = []

    # TODO support continuous random effects with monte carlo
    assert random_effects["group"] != "continuous"
    assert random_effects["individual"] != "continuous"

    with pyro.poutine.trace(param_only=True) as param_capture:
        loss_fn()
    params = [
        site["value"].unconstrained() for site in param_capture.trace.nodes.values()
    ]
    if optim == "sgd":
        optimizer = torch.optim.Adam(params, lr=lr)
    elif optim == "lbfgs":
        optimizer = torch.optim.LBFGS(params, lr=lr)
    else:
        raise ValueError("{} not supported optimizer".format(optim))

    if schedule:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=schedule, gamma=0.5
        )
        schedule_step_loss = False
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
        schedule_step_loss = True

    for t in range(args["timesteps"]):

        def closure():
            optimizer.zero_grad()
            loss = loss_fn()
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        scheduler.step(loss.item() if schedule_step_loss else t)
        losses.append(loss.item())
        print(
            "Loss: {}, AIC[{}]: ".format(loss.item(), t),
            2.0 * loss + 2.0 * num_parameters,
        )

    aic_final = 2.0 * losses[-1] + 2.0 * num_parameters
    print("AIC final: {}".format(aic_final))

    results = {}
    results["args"] = args
    results["sizes"] = config["sizes"]
    results["likelihoods"] = losses
    results["likelihood_final"] = losses[-1]
    results["aic_final"] = aic_final
    results["aic_num_parameters"] = num_parameters

    if args["resultsdir"] is not None and os.path.exists(args["resultsdir"]):
        re_str = "g" + (
            "n"
            if args["group"] is None
            else "d"
            if args["group"] == "discrete"
            else "c"
        )
        re_str += "i" + (
            "n"
            if args["individual"] is None
            else "d"
            if args["individual"] == "discrete"
            else "c"
        )
        results_filename = "expt_{}_{}_{}.json".format(
            args["dataset"], re_str, str(uuid.uuid4().hex)[0:5]
        )
        with open(os.path.join(args["resultsdir"], results_filename), "w") as f:
            json.dump(results, f)

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="seal", type=str)
    parser.add_argument("-g", "--group", default="none", type=str)
    parser.add_argument("-i", "--individual", default="none", type=str)
    parser.add_argument("-f", "--folder", default="./", type=str)
    parser.add_argument("-o", "--optim", default="sgd", type=str)
    parser.add_argument("-lr", "--learnrate", default=0.05, type=float)
    parser.add_argument("-t", "--timesteps", default=1000, type=int)
    parser.add_argument("-r", "--resultsdir", default="./results", type=str)
    parser.add_argument("-zi", "--zeroinflation", action="store_true")
    parser.add_argument("--seed", default=101, type=int)
    parser.add_argument("--truncate", default=-1, type=int)
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--schedule", default="", type=str)
    parser.add_argument("--validation", action="store_true")

    # sizes for generating fake data
    parser.add_argument("-ss", "--size-state", default=3, type=int)
    parser.add_argument("-sr", "--size-random", default=4, type=int)
    parser.add_argument("-sg", "--size-group", default=2, type=int)
    parser.add_argument("-si", "--size-individual", default=20, type=int)
    parser.add_argument("-st", "--size-timesteps", default=1800, type=int)
    args = parser.parse_args()

    if args.group == "none":
        args.group = None
    if args.individual == "none":
        args.individual = None

    run_expt(vars(args))
