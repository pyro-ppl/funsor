import math
from collections import OrderedDict
from functools import singledispatch

import torch
import torch.distributions as dist
from pyro.distributions.torch_distribution import MaskedDistribution
from pyro.distributions.util import broadcast_shape

from funsor.distributions import BernoulliLogits, MultivariateNormal, Normal
from funsor.domains import bint, reals
from funsor.gaussian import Gaussian, cholesky_solve
from funsor.terms import Independent
from funsor.torch import Tensor

# Conversion functions use fixed names for Pyro batch dims, but
# accept an event_inputs tuple for custom event dim names.
DIM_TO_NAME = tuple(map("_pyro_dim_{}".format, range(-100, 0)))
NAME_TO_DIM = dict(zip(DIM_TO_NAME, range(-100, 0)))


def tensor_to_funsor(tensor, event_inputs=(), event_output=0, dtype="real"):
    """
    Convert a :class:`torch.Tensor` to a :class:`funsor.torch.Tensor` .
    """
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(event_inputs, tuple)
    assert isinstance(event_output, int) and event_output >= 0
    inputs_shape = tensor.shape[:tensor.dim() - event_output]
    output_shape = tensor.shape[tensor.dim() - event_output:]
    dim_to_name = DIM_TO_NAME + event_inputs if event_inputs else DIM_TO_NAME

    # Squeeze shape of inputs.
    inputs = OrderedDict()
    squeezed_shape = []
    for dim, size in enumerate(inputs_shape):
        if size > 1:
            name = dim_to_name[dim - len(inputs_shape)]
            inputs[name] = bint(size)
            squeezed_shape.append(size)
    squeezed_shape = torch.Size(squeezed_shape)
    if squeezed_shape != inputs_shape:
        tensor = tensor.reshape(squeezed_shape + output_shape)

    return Tensor(tensor, inputs, dtype)


def funsor_to_tensor(funsor_, ndims, event_inputs=()):
    """
    Convert a :class:`funsor.torch.Tensor` to a :class:`torch.Tensor` .
    """
    assert isinstance(funsor_, Tensor)
    assert all(k.startswith("_pyro_dim_") or k in event_inputs for k in funsor_.inputs)
    name_to_dim = NAME_TO_DIM
    if event_inputs:
        dim_to_name = DIM_TO_NAME + event_inputs
        name_to_dim = dict(zip(dim_to_name, range(-len(dim_to_name), 0)))
    names = tuple(sorted(funsor_.inputs, key=name_to_dim.__getitem__))
    tensor = funsor_.align(names).data
    if names:
        # Unsqueeze shape of inputs.
        dims = list(map(name_to_dim.__getitem__, names))
        inputs_shape = [1] * (-dims[0])
        for dim, size in zip(dims, tensor.shape):
            inputs_shape[dim] = size
        inputs_shape = torch.Size(inputs_shape)
        tensor = tensor.reshape(inputs_shape + funsor_.output.shape)
    if ndims != tensor.dim():
        tensor = tensor.reshape((1,) * (ndims - tensor.dim()) + tensor.shape)
    assert tensor.dim() == ndims
    return tensor


def mvn_to_funsor(pyro_dist, event_dims=(), real_inputs=OrderedDict()):
    """
    Convert a joint :class:`torch.distributions.MultivariateNormal`
    distribution into a :class:`~funsor.terms.Funsor` with multiple real
    inputs.
    """
    assert isinstance(pyro_dist, torch.distributions.MultivariateNormal)
    assert isinstance(event_dims, tuple)
    assert isinstance(real_inputs, OrderedDict)
    loc = tensor_to_funsor(pyro_dist.loc, event_dims, 1)
    scale_tril = tensor_to_funsor(pyro_dist.scale_tril, event_dims, 2)
    precision = tensor_to_funsor(pyro_dist.precision_matrix, event_dims, 2)
    assert loc.inputs == scale_tril.inputs
    assert loc.inputs == precision.inputs
    info_vec = precision.data.matmul(loc.data.unsqueeze(-1)).squeeze(-1)
    log_prob = (-0.5 * loc.output.shape[0] * math.log(2 * math.pi)
                - scale_tril.data.diagonal(dim1=-1, dim2=-2).log().sum(-1)
                - 0.5 * (info_vec * loc.data).sum(-1))
    inputs = loc.inputs.copy()
    inputs.update(real_inputs)
    return Tensor(log_prob, loc.inputs) + Gaussian(info_vec, precision.data, inputs)


def matrix_and_mvn_to_funsor(matrix, mvn, event_dims=(), x_name="value_x", y_name="value_y"):
    """
    Convert a noisy affine function to a Gaussian. The noisy affine function is defined as::

        y = x @ matrix + mvn.sample()

    :param ~torch.Tensor matrix: A matrix with rightmost shape ``(x_size, y_size)``.
    :param ~torch.distributions.MultivariateNormal mvn: A multivariate normal
        distribution with ``event_shape == (y_size,)``.
    """
    assert isinstance(mvn, torch.distributions.MultivariateNormal)
    assert isinstance(matrix, torch.Tensor)
    x_size, y_size = matrix.shape[-2:]
    assert mvn.event_shape == (y_size,)
    info_vec = cholesky_solve(mvn.loc.unsqueeze(-1), mvn.scale_tril).squeeze(-1)
    log_prob = (-0.5 * y_size * math.log(2 * math.pi)
                - mvn.scale_tril.diagonal(dim1=-1, dim2=-2).log().sum(-1)
                - 0.5 * (info_vec * mvn.loc).sum(-1))

    batch_shape = broadcast_shape(matrix.shape[:-2], mvn.batch_shape)
    P_yy = mvn.precision_matrix.expand(batch_shape + (y_size, y_size))
    neg_P_xy = matrix.matmul(P_yy)
    P_xy = -neg_P_xy
    P_yx = P_xy.transpose(-1, -2)
    P_xx = neg_P_xy.matmul(matrix.transpose(-1, -2))
    precision = torch.cat([torch.cat([P_xx, P_xy], -1),
                           torch.cat([P_yx, P_yy], -1)], -2)
    info_y = info_vec.expand(batch_shape + (y_size,))
    info_x = -matrix.matmul(info_y.unsqueeze(-1)).squeeze(-1)
    info_vec = torch.cat([info_x, info_y], -1)

    info_vec = tensor_to_funsor(info_vec, event_dims, 1)
    precision = tensor_to_funsor(precision, event_dims, 2)
    inputs = info_vec.inputs.copy()
    inputs[x_name] = reals(x_size)
    inputs[y_name] = reals(y_size)
    return tensor_to_funsor(log_prob, event_dims) + Gaussian(info_vec.data, precision.data, inputs)


def matrix_and_mvn_to_funsor(matrix, mvn, event_dims=(), x_name="value_x", y_name="value_y"):
    """
    Convert a noisy affine function to a Gaussian. The noisy affine function is defined as::

        y = x @ matrix + mvn.sample()

    :param ~torch.Tensor matrix: A matrix with rightmost shape ``(x_dim, y_dim)``.
    :param ~torch.distributions.MultivariateNormal mvn: A multivariate normal
        distribution with ``event_shape == (y_dim,)``.
    """
    assert isinstance(mvn, torch.distributions.MultivariateNormal)
    assert isinstance(matrix, torch.Tensor)
    x_dim, y_dim = matrix.shape[-2:]
    assert mvn.event_shape == (y_dim,)
    batch_shape = broadcast_shape(matrix.shape[:-2], mvn.batch_shape)
    matrix = matrix.expand(batch_shape + (x_dim, y_dim))
    log_prob = (-0.5 * y_dim * math.log(2 * math.pi) -
                mvn.scale_tril.diagonal(dim1=-1, dim2=-2).log().sum(-1))
    mvn = mvn.expand(batch_shape)

    P_yy = mvn.precision_matrix
    neg_P_xy = matrix.matmul(P_yy)
    P_xy = -neg_P_xy
    P_yx = P_xy.transpose(-1, -2)
    P_xx = neg_P_xy.matmul(matrix.transpose(-1, -2))
    precision = torch.cat([torch.cat([P_xx, P_xy], -1),
                           torch.cat([P_yx, P_yy], -1)], -2)
    loc_y = mvn.loc
    loc_x = loc_y.new_zeros(batch_shape + (x_dim,))
    loc = torch.cat([loc_x, loc_y], -1)

    inputs = tensor_to_funsor(loc, event_dims, 1).inputs.copy()
    inputs[x_name] = reals(x_dim)
    inputs[y_name] = reals(y_dim)
    return tensor_to_funsor(log_prob, event_dims) + Gaussian(loc, precision, inputs)


@singledispatch
def dist_to_funsor(pyro_dist, event_inputs=()):
    """
    Convert a :class:`torch.distributions.Distribution` to a
    :class:`~funsor.terms.Funsor` .
    """
    assert isinstance(pyro_dist, torch.distributions.Distribution)
    raise ValueError("Cannot convert {} distribution to a Funsor"
                     .format(type(pyro_dist).__name__))


@dist_to_funsor.register(dist.Independent)
def _independent_to_funsor(pyro_dist, event_inputs=()):
    event_names = tuple("_event_{}".format(len(event_inputs) + i)
                        for i in range(pyro_dist.reinterpreted_batch_ndims))
    result = dist_to_funsor(pyro_dist.base_dist, event_inputs + event_names)
    for name in reversed(event_names):
        result = Independent(result, "value", name)
    return result


@dist_to_funsor.register(MaskedDistribution)
def _masked_to_funsor(pyro_dist, event_inputs=()):
    # FIXME This is subject to NANs.
    mask = tensor_to_funsor(pyro_dist._mask.float(), event_inputs)
    result = mask * dist_to_funsor(pyro_dist.base_dist, event_inputs)
    return result


@dist_to_funsor.register(dist.Categorical)
def _categorical_to_funsor(pyro_dist, event_inputs=()):
    return tensor_to_funsor(pyro_dist.logits, event_inputs + ("value",))


@dist_to_funsor.register(dist.Bernoulli)
def _bernoulli_to_funsor(pyro_dist, event_inputs=()):
    logits = tensor_to_funsor(pyro_dist.logits, event_inputs)
    return BernoulliLogits(logits)


@dist_to_funsor.register(dist.Normal)
def _normal_to_funsor(pyro_dist, event_inputs=()):
    loc = tensor_to_funsor(pyro_dist.loc, event_inputs)
    scale = tensor_to_funsor(pyro_dist.scale, event_inputs)
    return Normal(loc, scale)


@dist_to_funsor.register(dist.MultivariateNormal)
def _mvn_to_funsor(pyro_dist, event_inputs=()):
    loc = tensor_to_funsor(pyro_dist.loc, event_inputs, 1)
    scale_tril = tensor_to_funsor(pyro_dist.scale_tril, event_inputs, 2)
    return MultivariateNormal(loc, scale_tril)
