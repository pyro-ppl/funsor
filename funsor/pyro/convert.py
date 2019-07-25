from collections import OrderedDict

import pyro.distributions as dist
import torch

from funsor.distributions import BernoulliLogits, MultivariateNormal, Normal
from funsor.domains import bint
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


def dist_to_funsor(pyro_dist, event_inputs=()):
    """
    Convert a :class:`torch.distributions.Distribution` to a
    :class:`~funsor.terms.Funsor` .
    """
    assert isinstance(pyro_dist, torch.distributions.Distribution)

    if isinstance(pyro_dist, dist.Independent):
        event_names = tuple("_event_{}".format(len(event_inputs) + i)
                            for i in range(pyro_dist.reinterpreted_batch_ndims))
        result = dist_to_funsor(pyro_dist.base_dist, event_inputs + event_names)
        for name in reversed(event_names):
            result = Independent(result, "value", name)
        return result

    if isinstance(pyro_dist, dist.Categorical):
        return tensor_to_funsor(pyro_dist.logits, event_inputs + ("value",))

    if isinstance(pyro_dist, dist.Bernoulli):
        logits = tensor_to_funsor(pyro_dist.logits, event_inputs)
        return BernoulliLogits(logits)

    if isinstance(pyro_dist, dist.Normal):
        loc = tensor_to_funsor(pyro_dist.loc, event_inputs)
        scale = tensor_to_funsor(pyro_dist.scale, event_inputs)
        return Normal(loc, scale)

    if isinstance(pyro_dist, dist.MultivariateNormal):
        loc = tensor_to_funsor(pyro_dist.loc, event_inputs, 1)
        scale_tril = tensor_to_funsor(pyro_dist.scale_tril, event_inputs, 2)
        return MultivariateNormal(loc, scale_tril)

    raise ValueError("Cannot convert {} distribution to a Funsor"
                     .format(type(pyro_dist).__name__))
