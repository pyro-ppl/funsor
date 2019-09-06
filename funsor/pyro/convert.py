"""
This module follows a convention for converting between funsors and PyTorch
distribution objects. This convention is compatible with NumPy/PyTorch-style
broadcasting. Following PyTorch distributions (and Tensorflow distributions),
we consider "event shapes" to be on the right and broadcast-compatible "batch
shapes" to be on the left.

This module also aims to be forgiving in inputs and pedantic in outputs:
methods accept either the superclass :class:`torch.distributions.Distribution`
objects or the subclass :class:`pyro.distributions.TorchDistribution` objects.
Methods return only the narrower subclass
:class:`pyro.distributions.TorchDistribution` objects.
"""

import math
from collections import OrderedDict
from functools import singledispatch

import pyro.distributions
import torch
import torch.distributions as dist
from pyro.distributions.torch_distribution import MaskedDistribution
from pyro.distributions.util import broadcast_shape

from funsor.distributions import BernoulliLogits, MultivariateNormal, Normal
from funsor.domains import bint, reals
from funsor.gaussian import Gaussian, align_tensors, cholesky_solve
from funsor.interpreter import gensym
from funsor.joint import Joint
from funsor.terms import Funsor, Independent, Variable, eager
from funsor.torch import Tensor

# Conversion functions use fixed names for Pyro batch dims, but
# accept an event_inputs tuple for custom event dim names.
DIM_TO_NAME = tuple(map("_pyro_dim_{}".format, range(-100, 0)))
NAME_TO_DIM = dict(zip(DIM_TO_NAME, range(-100, 0)))


def tensor_to_funsor(tensor, event_inputs=(), event_output=0, dtype="real"):
    """
    Convert a :class:`torch.Tensor` to a :class:`funsor.torch.Tensor` .

    Note this should not touch data, but may trigger a
    :meth:`torch.Tensor.reshape` op.

    :param torch.Tensor tensor: A PyTorch tensor.
    :param tuple event_inputs: A tuple of names for rightmost tensor
        dimensions.  If ``tensor`` has these names, they will be converted to
        ``result.inputs``.
    :param int event_output: The number of tensor dimensions assigned to
        ``result.output``. These must be on the right of any ``event_input``
        dimensions.
    :return: A funsor.
    :rtype: funsor.torch.Tensor
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

    Note this should not touch data, but may trigger a
    :meth:`torch.Tensor.reshape` op.

    :param funsor.torch.Tensor funsor_: A funsor.
    :param int ndims: The number of result dims, ``== result.dim()``.
    :param tuple event_inputs: Names assigned to rightmost dimensions.
    :return: A PyTorch tensor.
    :rtype: torch.Tensor
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

    This should satisfy::

        sum(d.num_elements for d in real_inputs.values())
          == pyro_dist.event_shape[0]

    :param torch.distributions.MultivariateNormal pyro_dist: A
        multivariate normal distribution over one or more variables
        of real or vector or tensor type.
    :param tuple event_dims: A tuple of names for rightmost dimensions.
        These will be assigned to ``result.inputs`` of type ``bint``.
    :param OrderedDict real_inputs: A dict mapping real variable name
        to appropriately sized ``reals()``. The sum of all ``.numel()``
        of all real inputs should be equal to the ``pyro_dist`` dimension.
    :return: A funsor with given ``real_inputs`` and possibly additional
        bint inputs.
    :rtype: funsor.terms.Funsor
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


def funsor_to_mvn(gaussian, ndims, event_inputs=()):
    """
    Convert a :class:`~funsor.terms.Funsor` to a
    :class:`pyro.distributions.MultivariateNormal` , dropping the normalization
    constant.

    :param gaussian: A Gaussian funsor.
    :type gaussian: funsor.gaussian.Gaussian or funsor.joint.Joint
    :param int ndims: The number of batch dimensions in the result.
    :param tuple event_inputs: A tuple of names to assign to rightmost
        dimensions.
    :return: a multivariate normal distribution.
    :rtype: pyro.distributions.MultivariateNormal
    """
    assert sum(1 for d in gaussian.inputs.values() if d.dtype == "real") == 1
    if isinstance(gaussian, Joint):
        gaussian = gaussian.gaussian
    assert isinstance(gaussian, Gaussian)

    precision = gaussian.precision
    loc = gaussian.info_vec.unsqueeze(-1).cholesky_solve(precision.cholesky()).squeeze(-1)

    int_inputs = OrderedDict((k, d) for k, d in gaussian.inputs.items() if d.dtype != "real")
    loc = Tensor(loc, int_inputs)
    precision = Tensor(precision, int_inputs)
    assert len(loc.output.shape) == 1
    assert precision.output.shape == loc.output.shape * 2

    loc = funsor_to_tensor(loc, ndims + 1, event_inputs)
    precision = funsor_to_tensor(precision, ndims + 2, event_inputs)
    return pyro.distributions.MultivariateNormal(loc, precision_matrix=precision)


def funsor_to_cat_and_mvn(funsor_, ndims, event_inputs):
    """
    Converts a labeled gaussian mixture model to a pair of distributions.

    :param funsor.joint.Joint funsor_: A Gaussian mixture funsor.
    :param int ndims: The number of batch dimensions in the result.
    :return: A pair ``(cat, mvn)``, where ``cat`` is a
        :class:`~pyro.distributions.Categorical` distribution over mixture
        components and ``mvn`` is a
        :class:`~pyro.distributions.MultivariateNormal` with rightmost batch
        dimension ranging over mixture components.
    """
    assert isinstance(funsor_, Joint), funsor_
    assert sum(1 for d in funsor_.inputs.values() if d.dtype == "real") == 1
    assert event_inputs, "no components name found"
    assert not funsor_.deltas
    discrete = funsor_.discrete
    gaussian = funsor_.gaussian
    assert isinstance(discrete, Tensor)
    assert isinstance(gaussian, Gaussian)

    logits = funsor_to_tensor(discrete + gaussian.log_normalizer, ndims + 1, event_inputs)
    cat = pyro.distributions.Categorical(logits=logits)
    mvn = funsor_to_mvn(gaussian, ndims + 1, event_inputs)
    assert cat.batch_shape == mvn.batch_shape[:-1]
    return cat, mvn


class AffineNormal(Funsor):
    """
    Represents a conditional diagonal normal distribution over a random
    variable ``Y`` whose mean is an affine function of a random variable ``X``.
    The likelihood of ``X`` is thus::

        AffineNormal(matrix, loc, scale).condition(y).log_density(x)

    which is equivalent to::

        Normal(x @ matrix + loc, scale).to_event(1).log_prob(y)

    :param ~funsor.terms.Funsor matrix: A transformation from ``X`` to ``Y``.
        Should have rightmost shape ``(x_dim, y_dim)``.
    :param ~funsor.terms.Funsor loc: A constant offset for ``Y``'s mean.
        Should have output shape ``(y_dim,)``.
    :param ~funsor.terms.Funsor scale: Standard deviation for ``Y``.
        Should have output shape ``(y_dim,)``.
    :param ~funsor.terms.Funsor value_x: A value ``X``.
    :param ~funsor.terms.Funsor value_y: A value ``Y``.
    """
    def __init__(self, matrix, loc, scale, value_x, value_y):
        assert len(matrix.output.shape) == 2
        assert value_x.output == reals(matrix.output.shape[0])
        assert value_y.output == reals(matrix.output.shape[1])
        inputs = OrderedDict()
        for f in (matrix, loc, scale, value_x, value_y):
            inputs.update(f.inputs)
        output = reals()
        super().__init__(inputs, output)
        self.matrix = matrix
        self.loc = loc
        self.scale = scale
        self.value_x = value_x
        self.value_y = value_y


@eager.register(AffineNormal, Tensor, Tensor, Tensor, Tensor, (Funsor, Tensor))
def eager_affine_normal(matrix, loc, scale, value_x, value_y):
    assert len(matrix.output.shape) == 2
    assert value_x.output == reals(matrix.output.shape[0])
    assert value_y.output == reals(matrix.output.shape[1])
    tensors = (matrix, loc, scale, value_x)
    int_inputs, tensors = align_tensors(*tensors)
    matrix, loc, scale, value_x = tensors

    loc = loc + value_x.unsqueeze(-2).matmul(matrix).squeeze(-2)
    i_name = gensym("i")
    y_name = gensym("y")
    y_i_name = gensym("y_i")
    int_inputs[i_name] = bint(value_y.output.shape[0])
    loc, scale = torch.broadcast_tensors(loc, scale)
    loc = Tensor(loc, int_inputs)
    scale = Tensor(scale, int_inputs)
    y_dist = Independent(Normal(loc, scale, y_i_name), y_name, i_name, y_i_name)
    return y_dist(**{y_name: value_y})


@eager.register(AffineNormal, Tensor, Tensor, Tensor, Funsor, Tensor)
def eager_affine_normal(matrix, loc, scale, value_x, value_y):
    assert len(matrix.output.shape) == 2
    assert value_x.output == reals(matrix.output.shape[0])
    assert value_y.output == reals(matrix.output.shape[1])
    tensors = (matrix, loc, scale, value_y)
    int_inputs, tensors = align_tensors(*tensors)
    matrix, loc, scale, value_y = tensors

    assert value_y.size(-1) == loc.size(-1)
    prec_sqrt = matrix / scale.unsqueeze(-2)
    precision = prec_sqrt.matmul(prec_sqrt.transpose(-1, -2))
    delta = (value_y - loc) / scale
    info_vec = prec_sqrt.matmul(delta.unsqueeze(-1)).squeeze(-1)
    log_normalizer = (-0.5 * loc.size(-1) * math.log(2 * math.pi)
                      - 0.5 * delta.pow(2).sum(-1) - scale.log().sum(-1))
    precision = precision.expand(info_vec.shape + (-1,))
    log_normalizer = log_normalizer.expand(info_vec.shape[:-1])
    inputs = int_inputs.copy()
    x_name = gensym("x")
    inputs[x_name] = value_x.output
    x_dist = Tensor(log_normalizer, int_inputs) + Gaussian(info_vec, precision, inputs)
    return x_dist(**{x_name: value_x})


def matrix_and_mvn_to_funsor(matrix, mvn, event_dims=(), x_name="value_x", y_name="value_y"):
    """
    Convert a noisy affine function to a Gaussian. The noisy affine function is
    defined as::

        y = x @ matrix + mvn.sample()

    The result is a non-normalized Gaussian funsor with two real inputs,
    ``x_name`` and ``y_name``, corresponding to a conditional distribution of
    real vector ``y` given real vector ``x``.

    :param torch.Tensor matrix: A matrix with rightmost shape ``(x_size, y_size)``.
    :param mvn: A multivariate normal distribution with
        ``event_shape == (y_size,)``.
    :type mvn: torch.distributions.MultivariateNormal or
        torch.distributions.Independent of torch.distributions.Normal
    :param tuple event_dims: A tuple of names for rightmost dimensions.
        These will be assigned to ``result.inputs`` of type ``bint``.
    :param str x_name: The name of the ``x`` random variable.
    :param str y_name: The name of the ``y`` random variable.
    :return: A funsor with given ``real_inputs`` and possibly additional
        bint inputs.
    :rtype: funsor.terms.Funsor
    """
    assert (isinstance(mvn, torch.distributions.MultivariateNormal) or
            (isinstance(mvn, torch.distributions.Independent) and
             isinstance(mvn.base_dist, torch.distributions.Normal)))
    assert isinstance(matrix, torch.Tensor)
    x_size, y_size = matrix.shape[-2:]
    assert mvn.event_shape == (y_size,)

    # Handle diagonal normal distributions as an efficient special case.
    if isinstance(mvn, torch.distributions.Independent):
        return AffineNormal(tensor_to_funsor(matrix, event_dims, 2),
                            tensor_to_funsor(mvn.base_dist.loc, event_dims, 1),
                            tensor_to_funsor(mvn.base_dist.scale, event_dims, 1),
                            Variable(x_name, reals(x_size)),
                            Variable(y_name, reals(y_size)))

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


@singledispatch
def dist_to_funsor(pyro_dist, event_inputs=()):
    """
    Convert a PyTorch distribution to a Funsor.

    This is currently implemented for only a subset of distribution types.

    :param torch.distribution.Distribution: A PyTorch distribution.
    :return: A funsor.
    :rtype: funsor.terms.Funsor
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
        result = Independent(result, "value", name, "value")
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
