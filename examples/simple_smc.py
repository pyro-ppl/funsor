import torch

import funsor
from funsor.distributions import Normal
from funsor.domains import reals
from funsor.interpreter import interpretation
from funsor.montecarlo import sequential_monte_carlo
from funsor.terms import Variable


def main(args):

   with interpretation(sequential_monte_carlo):
        px = Normal(0, 1, 'x')
        fx = (Variable('x', reals()) - 0.5) ** 2
        py = Normal('x', 1, 'y')
        fy = (Variable('y', reals()) - 0.3) ** 2

        loss = Integrate(px, fx + Integrate(py, fy, 'y'), 'x')
