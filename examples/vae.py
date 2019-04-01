from __future__ import absolute_import, division, print_function

import argparse
import os
import pdb
from collections import OrderedDict

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

import funsor
import funsor.distributions as dist
import funsor.ops as ops
from funsor.domains import bint, reals

REPO_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_PATH, 'data')


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)

    def forward(self, image):
        image = image.reshape(image.shape[:-2] + (-1,))
        h1 = F.relu(self.fc1(image))
        loc = self.fc21(h1)
        scale = self.fc22(h1)
        return loc, scale


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def forward(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))


def main(args):
    encoder = Encoder()
    decoder = Decoder()

    encode = funsor.torch.function(reals(28, 28), (reals(20), reals(20)))(encoder)
    decode = funsor.torch.function(reals(20), reals(28, 28))(decoder)

    @funsor.interpreter.interpretation(funsor.montecarlo.monte_carlo)
    def loss_function(data, subsample_scale):
        # Lazily sample from the guide.
        loc, scale = encode(data)
        q = funsor.Independent(
            dist.Normal(loc['i'], scale['i'], value='z'),
            'z', 'i')

        # Evaluate the model likelihood at the lazy value z.
        probs = decode('z')
        p = dist.Bernoulli(probs['x', 'y'], value=data['x', 'y'])
        p = p.reduce(ops.add, frozenset(['x', 'y']))

        # Construct an elbo. This is where sampling happens.
        elbo = funsor.Integrate(q, p - q, frozenset(['z']))
        elbo = elbo.reduce(ops.add, 'batch') * subsample_scale
        loss = -elbo
        return loss

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_PATH, train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True)

    encoder.train()
    decoder.train()
    optimizer = optim.Adam(list(encoder.parameters()) +
                           list(decoder.parameters()), lr=1e-3)
    for epoch in range(args.num_epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            subsample_scale = float(len(train_loader.dataset) / len(data))
            data = data[:, 0, :, :]
            data = funsor.Tensor(data, OrderedDict(batch=bint(len(data))))

            optimizer.zero_grad()
            loss = loss_function(data, subsample_scale)
            assert isinstance(loss, funsor.torch.Tensor), loss.pretty()
            loss.data.backward()
            train_loss += loss.item()
            optimizer.step()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('-n', '--num-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--pdb', action='store_true')
    args = parser.parse_args()

    if args.pdb:
        try:
            main(args)
        except Exception as e:
            print(repr(e))
            torch.set_printoptions(threshold=10, edgeitems=1)
            pdb.post_mortem()
    else:
        main(args)
