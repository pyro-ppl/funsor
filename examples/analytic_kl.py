from __future__ import absolute_import, division, print_function

import pyro.distributions as dist
import torch

import funsor
import funsor.minipyro as pyro
import funsor.distributions as dist


def model(data):
    x = pyro.sample("x", dist.Normal(0, 1))
    y = pyro.sample("y", dist.Normal(x, 1))
    pyro.sample("z", dist.Normal(y, 1), obs=data)


def guide_1(data):
    x = pyro.sample("x", dist.Normal(0, 1))
    y = pyro.sample("y", dist.Normal(x, 1))
    return x, y


def guide_2(data):
    y = pyro.sample("y", dist.Normal(0, 1))
    x = pyro.sample("x", dist.Normal(y, 1))
    return x, y


def elbos(z):
    x = Variable('x', reals())
    y = Variable('x', reals())
    z = Tensor(torch.randn(()))

    px = dist.Normal(0, 1, value=x)
    py = dist.Normal(nonlinear(x), 1, value=y)
    pz = dist.Normal(nonlinear(y), 1, value=z)
    p = px + py + pz

    q1x = dist.Normal(0, 1, value=x)
    q1y = dist.Normal(nonlinear(x), 1, value=y)
    q1 = q1x + q1y
    elbo1 = Integrate(q1, p - q1, frozenset(['x', 'y']))
    """
    Integrate(q1, p - q1, {'x', 'y'})
    --> Integrate(q1x + q1y, px + py + pz - (q1x + q1y), {'x', 'y'})

    --> Integrate(q1x + q1y, px - q1x, {'x', 'y'})  # only on x
      + Integrate(q1x + q1y, py + pz - q1y, {'x', 'y'})  # on x and y

    --> Integrate(q1x, px - q1x, {'x', 'y'})               # analytic
      + Integrate(q1x + q1y, py + pz - q1y, {'x', 'y'})    # monte carlo x

    --> Tensor(...)
      + Integrate(Delta(...) + q1y, py + pz - q1y, {'x', 'y'})

    --> Tensor(...)
      + Integrate(q1y_given_x, py_given_x + pz - q1y_given_x, {'y'})

    --> Tensor(...)
      + Integrate(q1y_given_x, py_given_x - q1y_given_x, {'y'})  # analytic
      + Integrate(q1y_given_x, pz, {'y'})  # monte carlo

    --> Tensor(...)
      + Tensor(...)
      + Integrate(Delta, pz, {'y'})

    --> Tensor(...)
      + Tensor(...)
      + Tensor(...)

    --> Tensor(...)
    ---------------------
    optimal = Integrate(q1x, px - q1x, {'x', 'y'})  # analytic
            + Integrate(q1x + q1y, py - q1y, {'y', 'x'})  # monte carlo x
            + Integrate(q1x + q1y, pz, {'x', 'y'})  # monte carlo x
    """

    q2y = dist.Normal(0, 1, value=y)
    q2x = dist.Normal(nonlinear(y), 1, value=x)
    q2 = q2x + q2y
    elbo2 = Integrate(q2, p - q2, frozenset(['x', 'y']))
    """
    Integrate(q2, p - q2, {'x', 'y'})
    --> Integrate(q2x + q2y, px + py + pz - (q2x + q2y), {'x', 'y'})

    --> Integrate(q2x + q2y, px, {'x', 'y'})  # only on x
      + Integrate(q2x + q2y, py + q2x, {'x', 'y'})  # on x,y
      + Integrate(q2x + q2y, pz - q2y, {'x', 'y'})  # only on y

    --> Integrate(q2x + q2y, px, {'x', 'y'})  # monte carlo y
      + Integrate(q2x + q2y, py + q2x, {'x', 'y'})  # monte carlo y
      + Integrate(q2x + q2y, pz - q2y, {'x', 'y'})  # drop x

    --> Integrate(q2x + Delta(...), px, {'x'})
      + Integrate(q2x + Delta(...), py + q2x, {'x', 'y'})
      + Integrate(q2y, pz - q2y, {'y'})  # split

    --> Integrate(q2x_given_y, px, {'x'})  # analytic
      + Integrate(q2x_given_y, py_given_y + q2x_given_y, {'x'}) # split
      + Integrate(q2y, pz, {'y'})  # monte carlo
      - Integrate(q2y, q2y, {'y'})  # analytic entropy

    --> Tensor(...)
      + Integrate(q2x_given_y, py_given_y, {'x'}) # monte carlo
      + Integrate(q2x_given_y, q2x_given_y, {'x'}) # analytic entropy
      + Integrate(Delta(...), pz, {'y'})
      + Tensor(...)

    --> Tensor(...)
      + Integrate(Delta(...), py_given_y, {'x'})
      + Tensor(...)
      + Tensor(...)
      + Tensor(...)

    --> Tensor(...)
    """
