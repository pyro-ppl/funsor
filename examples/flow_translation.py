# This is a (currently) non-functioning translation of the code in Stefan's
# tutorial on normalizing flows in Pyro:
# http://pyro.ai/examples/normalizing_flows_i.html

# cell 1
import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T

import funsor

funsor.set_backend("torch")


# ------------------ cell 2
# dx = funsor.Lebesgue('x')
dist_x = funsor.torch.distributions.Normal(0., 1., 'x')  # + dx
exp_transform = funsor.ops.exp(funsor.Variable('y', funsor.reals()))


# ------------------ cell 3
dist_y = dist_x(x=exp_transform)  # how it should work, with Lebesgue term


# cell 4 (roughly..)
d_samples_x = dist_x.sample('x', OrderedDict(i=funsor.bint(1000)))
# TODO print mean or plot histogram
d_samples_y = dist_y.sample('y', OrderedDict(i=funsor.bint(1000)))
# TODO print mean or plot histogram


# ------------------ cell 5
dist_x = funsor.torch.distributions.Normal(0., 1., 'x')
# more literal
affine_transform = 3. + 0.5 * funsor.Variable('y', funsor.reals())
exp_transform = funsor.ops.exp(affine_transform)
dist_y = dist_x(x=exp_transform)
# inlined/idiomatic
y = funsor.Variable('y', funsor.reals())
dist_y = dist_x(x=(3. + 0.5 * y).exp())

d_samples_x = dist_x.sample('x', OrderedDict(i=funsor.bint(1000)))
# TODO print mean or plot histogram
d_samples_y = dist_y.sample('y', OrderedDict(i=funsor.bint(1000)))
# TODO print mean or plot histogram


# ------------------ cell 6 
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
# define data, skip plots for now
n_samples = 1000
X, y = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
X = StandardScaler().fit_transform(X)


# -------------------- cell 7
base_dist = funsor.torch.distributions.Normal(torch.zeros(2), torch.ones(2), 'x')
spline_transform = funsor.ops.SplineOp(2, count_bins=16)(
    funsor.Variable('y', funsor.reals(2)),
    funsor.Variable('theta', funsor.reals(2))
)
flow_dist = base_dist(x=spline_transform)


# -------------------- cell 8
# assert X.shape == (100, 2)
data_funsor = funsor.Tensor(X, OrderedDict(i=funsor.bint(100)))
loss = -flow_dist(y=data_funsor).reduce(ops.add, frozenset(data_funsor.inputs))

# the reality
theta = torch.rand(2, requires_grad=True)
optimizer = torch.optim.Adam([theta])
for step in range(steps):
    optimizer.zero_grad()
    loss(theta=theta).data.backward()
    optimizer.step()

# the dream
with funsor.interpreter.interpretation(adam(config)):
    with funsor.adjoint.AdjointTape() as tape:
        loss_expr = loss.reduce(ops.min, 'theta')
    theta = tape.adjoint(ops.min. ops.add, loss_expr, (theta0,))['theta']

# the dream within the dream
@adam.register(funsor.terms.Reduce, funsor.ops.MinOp, funsor.Funsor, frozenset)
def adam_min(op, arg, reduced_vars):
    params = {var: torch.randn(*arg.inputs[var], requires_grad=True)
              for var in reduced_vars.intersection(arg.inputs)}
    optimizer = torch.optim.Adam(list(params.values()))
    for step in range(steps):
        optimizer.zero_grad()
        loss(**params).data.backward()
        optimizer.step()
    return loss(**params)


# ------------------ skipping ahead to cell 13
dist_base = funsor.torch.distributions.Normal(0., 1., 'x')
dist_x1 = dist_base(x=funsor.ops.SplineOp(1)(funsor.Variable('x1', funsor.reals(1))))

x2_transform = funsor.ops.SplineOp(1)(
    funsor.Variable('y', funsor.reals(1)),
    funsor.Variable('theta', funsor.reals(1))
)

@funsor.function(funsor.reals(1, 1), funsor.reals(1), funsor.reals(1), funsor.reals(1))
def theta_nn(weight, bias, x):
    return weight @ x + bias


assert isinstance(theta_nn, funsor.Funsor)
assert set(theta_nn.inputs) == {"weight", "bias", "x"}

dist_x2_given_x1 = dist_base(x=x2_transform, theta=theta_nn)

dist_x = dist_x1 + dist_x2_given_x1(y='x2', x='x1')

loss = -dist_x(x1=x1_data, x2=x2_data).reduce(
    ops.add, frozenset(x1_data.inputs) | frozenset(x2_data.inputs))

weight = torch.randn(1, 1, requires_grad=True)
bias = torch.randn(1, requires_grad=True)
optimizer = torch.optim.Adam([weight, bias])
for step in range(steps):
    optimizer.zero_grad()
    loss(weight=weight, bias=bias).data.backward()
    optimizer.step()


# ------------------- cell 17
samples_joint_x1_x2 = dist_x.sample({"x1", "x2"}, OrderedDict(i=funsor.bint(10)))
samples_x2_given_x1 = dist_x(x1=x1_data).sample("x2", OrderedDict(i=funsor.bint(10)))


# ---------------- alternate version of nn
dist_base = funsor.torch.distributions.Normal(0., 1., 'x')
dist_x1 = dist_base(x=funsor.ops.SplineOp(1)(
    funsor.Variable('x1', funsor.reals(1)),
    funsor.Variable('theta1', funsor.reals(1))
))

x2_transform = funsor.ops.SplineOp(1)(
    funsor.Variable('y', funsor.reals(1)),
    funsor.Variable('theta', funsor.reals(1))
)

theta_nn = funsor.function(funsor.reals(1), funsor.reals(1))(
    torch.nn.Linear(1, 1))('x')

dist_x2_given_x1 = dist_base(x=x2_transform(theta=theta_nn))
dist_x = dist_x1 + dist_x2_given_x1(y='x2', x='x1')

theta1 = torch.randn(2, requires_grad=True)
optimizer = torch.optim.Adam(...)
for step in range(steps):
    optimizer.zero_grad()
    # substituting data triggers neural net evaluation
    loss = -dist_x(theta1=theta1, x1=x1_data, x2=x2_data).reduce(ops.add)
    loss.data.backward()
    optimizer.step()
