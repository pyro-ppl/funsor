from __future__ import absolute_import, division, print_function

import argparse

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

import funsor
import funsor.ops as ops
import funsor.distributions as dist
from funsor.domains import reals


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)

    def forward(self, image):
        h1 = F.relu(self.fc1(image))
        return self.fc21(h1), self.fc22(h1)


class Decoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def forward(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def old_forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def main(args):
    encoder = Encoder()
    decoder = Decoder()

    @funsor.function(reals(28, 28), reals(2, 20))
    def encode(image):
        loc, scale = encoder(image)
        return torch.stack([loc, scale], dim=-2)

    @funsor.function(reals(28, 20), reals(20))
    def decode(z):
        return decoder(z)

    @funsor.interpreter.interpretation(funsor.interpreter.monte_carlo)
    def loss_function(data):
        loc, scale = encode(data)
        z = funsor.Variable('z', reals(20))
        q = dist.Normal(loc, scale, value=z)

        probs = decode(z)
        p = dist.Bernoulli(probs, value=data)

        elbo = (q.exp() * (p - q)).reduce(ops.add)
        loss = -elbo
        return loss.data

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True)

    encoder.train()
    decoder.train()
    optimizer = optim.Adam(encoder.parameters() +
                           decoder.parameters(), lr=1e-3)
    for epoch in range(args.num_epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = loss_function(data)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    main(args)
