import torch

import funsor
import funsor.distributions as dist
from funsor.terms import Number


def model():
    component = dist.Categorical(torch.tensor([0.5, 0.5]))(value='component')
    component1 = funsor.Delta("x", Number(0.0)) + dist.Normal(-1, 1)(value='y')
    component2 = dist.Normal(0.0, 1.0)(value='x') + dist.Normal(1, 1)(value='y')
    joint = component + funsor.Stack("component", (component1, component2))
    # components = funsor.Stack("component", (funsor.Delta("x", Number(0.0)), dist.Normal(0.0, 1.0)(value='x')))
    # log_prob = (components + component).reduce(ops.logaddexp, "component")
    ll = joint(x=0)
    print("ll", ll)
