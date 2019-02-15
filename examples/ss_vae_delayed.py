from __future__ import absolute_import, division, print_function

import argparse

import torch
import torch.nn as nn

import funsor
import funsor.minipyro as pyro
import pyro.distributions as dist


class Decoder(nn.Module):
    pass  # TODO


class NuisanceEncoder(nn.Module):
    pass  # TODO


class SalientEncoder(nn.Module):
    pass  # TODO


decoder = funsor.function((), (), ())(Decoder())
nuisance_encoder = funsor.function((), ('loc_scale',))(NuisanceEncoder())
salient_encoder = funsor.function((), (), ())(SalientEncoder())


def model(image=None):
    pyro.module("decoder", decoder)
    nuisance = pyro.sample("nuisance", dist.Normal(0., 1.))
    salient = pyro.sample("salient", dist.Normal(0., 1.))
    image = pyro.sample("image", dist.Beta(decoder(nuisance, salient)),
                        obs=image)
    return image


def guide(image):
    pyro.module("nuisance_encoder", nuisance_encoder)
    pyro.module("salient_encoder", salient_encoder)

    # This sample statement is delayed.
    salient = pyro.sample("salient", dist.Normal(salient_encoder(image)))

    # Now we can draw samples before passing through the encoder.
    loc, scale = nuisance_encoder(image, salient)
    nuisance = pyro.sample("nuisance", dist.Normal(loc, scale))
    return salient, nuisance


def main(args):
    # Generate fake data.
    data = funsor.Tensor(('data',), torch.randn(100))

    # Train.
    optim = pyro.Adam({'lr': args.learning_rate})
    svi = pyro.SVI(model, pyro.deferred(guide), optim, pyro.elbo)
    for step in range(args.steps):
        svi.step(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Kalman filter example")
    parser.add_argument("-n", "--steps", default=101, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("--eager", action='store_true')
    parser.add_argument("--filter", action='store_true')
    parser.add_argument("--xfail-if-not-implemented", action='store_true')
    args = parser.parse_args()

    if args.xfail_if_not_implemented:
        try:
            main(args)
        except NotImplementedError:
            print('XFAIL')
    else:
        main(args)
