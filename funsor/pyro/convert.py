from collections import OrderedDict

import pyro.distributions as dist
import torch

from funsor.domains import bint
from funsor.torch import Tensor

# Conversion functions use fixed names for Pyro batch dims.
DIM_TO_NAME = tuple(map("_pyro_dim_{}".format, range(-100, 0)))
NAME_TO_DIM = dict(zip(DIM_TO_NAME, range(-100, 0)))


def tensor_to_funsor(tensor, event_dim=0, dtype="real"):
    """
    Convert a :class:`torch.Tensor` to a :class:`funsor.torch.Tensor` .
    """
    assert isinstance(tensor, torch.Tensor)
    batch_shape = tensor.shape[:tensor.dim() - event_dim]
    event_shape = tensor.shape[tensor.dim() - event_dim:]

    # Squeeze batch_shape.
    inputs = OrderedDict()
    squeezed_batch_shape = []
    for dim, size in enumerate(batch_shape):
        if size > 1:
            name = DIM_TO_NAME[dim - len(batch_shape)]
            inputs[name] = bint(size)
            squeezed_batch_shape.append(size)
    squeezed_batch_shape = torch.Size(squeezed_batch_shape)
    if squeezed_batch_shape != batch_shape:
        batch_shape = squeezed_batch_shape
        tensor = tensor.reshape(batch_shape + event_shape)

    return Tensor(tensor, inputs, dtype)


def funsor_to_tensor(funsor_, ndims):
    """
    Convert a :class:`funsor.torch.Tensor` to a :class:`torch.Tensor` .
    """
    assert isinstance(funsor_, Tensor)
    assert all(k.startswith("_pyro_dim_") for k in funsor_.inputs)
    names = tuple(sorted(funsor_.inputs, key=NAME_TO_DIM.__getitem__))
    tensor = funsor_.align(names).data
    if names:
        # Unsqueeze batch_shape.
        dims = list(map(NAME_TO_DIM.__getitem__, names))
        batch_shape = [1] * (-dims[0])
        for dim, size in zip(dims, tensor.shape):
            batch_shape[dim] = size
        batch_shape = torch.Size(batch_shape)
        tensor = tensor.reshape(batch_shape + funsor_.output.shape)
    if ndims != tensor.dim():
        tensor = tensor.reshape((1,) * (ndims - tensor.dim()) + tensor.shape)
    assert tensor.dim() == ndims
    return tensor


def dist_to_funsor(pyro_dist, reinterpreted_batch_ndims=0):
    """
    Convert a :class:`torch.distributions.Distribution` to a
    :class:`~funsor.terms.Funsor` .
    """
    assert isinstance(pyro_dist, torch.distributions.Distribution)
    while isinstance(pyro_dist, dist.Independent):
        reinterpreted_batch_ndims += pyro_dist.reinterpreted_batch_ndims
        pyro_dist = pyro_dist.base_dist
    event_dim = pyro_dist.event_dim + reinterpreted_batch_ndims

    if isinstance(pyro_dist, dist.Categorical):
        return tensor_to_funsor(pyro_dist.logits, event_dim=event_dim + 1)
    if isinstance(pyro_dist, dist.Normal):
        raise NotImplementedError("TODO")
    if isinstance(pyro_dist, dist.MultivariateNormal):
        raise NotImplementedError("TODO")

    raise ValueError("Cannot convert {} distribution to a Funsor"
                     .format(type(pyro_dist).__name__))
