# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse

import torch
from pyroapi import distributions as dist
from pyroapi import infer, optim, pyro, pyro_backend
from torch.distributions import constraints

import funsor
from funsor.interpreter import interpretation
from funsor.montecarlo import monte_carlo


def main(args):
    funsor.set_backend("torch")

    # Define a basic model with a single Normal latent random variable `loc`
    # and a batch of Normally distributed observations.
    def model(data):
        loc = pyro.sample("loc", dist.Normal(0., 1.))
        with pyro.plate("data", len(data), dim=-1):
            pyro.sample("obs", dist.Normal(loc, 1.), obs=data)

    # Define a guide (i.e. variational distribution) with a Normal
    # distribution over the latent random variable `loc`.
    def guide(data):
        guide_loc = pyro.param("guide_loc", torch.tensor(0.))
        guide_scale = pyro.param("guide_scale", torch.tensor(1.),
                                 constraint=constraints.positive)
        pyro.sample("loc", dist.Normal(guide_loc, guide_scale))

    # Generate some data.
    torch.manual_seed(0)
    data = torch.randn(100) + 3.0

    # Because the API in minipyro matches that of Pyro proper,
    # training code works with generic Pyro implementations.
    with pyro_backend(args.backend), interpretation(monte_carlo):
        # Construct an SVI object so we can do variational inference on our
        # model/guide pair.
        Elbo = infer.JitTrace_ELBO if args.jit else infer.Trace_ELBO
        elbo = Elbo()
        adam = optim.Adam({"lr": args.learning_rate})
        svi = infer.SVI(model, guide, adam, elbo)

        # Basic training loop
        pyro.get_param_store().clear()
        for step in range(args.num_steps):
            loss = svi.step(data)
            if args.verbose and step % 100 == 0:
                print("step {} loss = {}".format(step, loss))

        # Report the final values of the variational parameters
        # in the guide after training.
        if args.verbose:
            for name in pyro.get_param_store():
                value = pyro.param(name).data
                print("{} = {}".format(name, value.detach().cpu().numpy()))

        # For this simple (conjugate) model we know the exact posterior. In
        # particular we know that the variational distribution should be
        # centered near 3.0. So let's check this explicitly.
        assert (pyro.param("guide_loc") - 3.0).abs() < 0.1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minipyro demo")
    parser.add_argument("-b", "--backend", default="funsor")
    parser.add_argument("-n", "--num-steps", default=1001, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.02, type=float)
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
