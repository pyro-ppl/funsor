# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: VAE MNIST
==================

"""

import argparse
import os
import typing
from collections import OrderedDict

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.datasets import MNIST

import funsor
import funsor.ops as ops
import funsor.torch.distributions as dist
from funsor.domains import Bint, Reals

REPO_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_PATH, "data")


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)

    def forward(self, image: Reals[28, 28]) -> typing.Tuple[Reals[20], Reals[20]]:
        image = image.reshape(image.shape[:-2] + (-1,))
        h1 = F.relu(self.fc1(image))
        loc = self.fc21(h1)
        scale = self.fc22(h1).exp()
        return loc, scale


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def forward(self, z: Reals[20]) -> Reals[28, 28]:
        h3 = F.relu(self.fc3(z))
        out = torch.sigmoid(self.fc4(h3))
        return out.reshape(out.shape[:-1] + (28, 28))


def main(args):
    funsor.set_backend("torch")

    # XXX Temporary fix after https://github.com/pyro-ppl/pyro/pull/2701
    import pyro

    pyro.enable_validation(False)

    encoder = Encoder()
    decoder = Decoder()

    # These rely on type hints on the .forward() methods.
    encode = funsor.function(encoder)
    decode = funsor.function(decoder)

    @funsor.montecarlo.MonteCarlo()
    def loss_function(data, subsample_scale):
        # Lazily sample from the guide.
        loc, scale = encode(data)
        q = funsor.Independent(
            dist.Normal(loc["i"], scale["i"], value="z_i"), "z", "i", "z_i"
        )

        # Evaluate the model likelihood at the lazy value z.
        probs = decode("z")
        p = dist.Bernoulli(probs["x", "y"], value=data["x", "y"])
        p = p.reduce(ops.add, {"x", "y"})

        # Construct an elbo. This is where sampling happens.
        elbo = funsor.Integrate(q, p - q, "z")
        elbo = elbo.reduce(ops.add, "batch") * subsample_scale
        loss = -elbo
        return loss

    train_loader = torch.utils.data.DataLoader(
        MNIST(DATA_PATH, train=True, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size,
        shuffle=True,
    )

    encoder.train()
    decoder.train()
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3
    )
    for epoch in range(args.num_epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            subsample_scale = float(len(train_loader.dataset) / len(data))
            data = data[:, 0, :, :]
            data = funsor.Tensor(data, OrderedDict(batch=Bint[len(data)]))

            optimizer.zero_grad()
            loss = loss_function(data, subsample_scale)
            assert isinstance(loss, funsor.Tensor), loss.pretty()
            loss.data.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % 50 == 0:
                print(f"  loss = {loss.item()}")
                if batch_idx and args.smoke_test:
                    return
        print(f"epoch {epoch} train_loss = {train_loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE MNIST Example")
    parser.add_argument("-n", "--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    main(args)
