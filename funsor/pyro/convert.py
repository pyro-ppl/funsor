# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

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

import torch

import funsor.ops as ops
from funsor.cnf import Contraction
from funsor.delta import Delta
from funsor.domains import Array, Bint, Real, Reals
from funsor.gaussian import Gaussian
from funsor.interpreter import gensym
from funsor.tensor import Tensor
from funsor.terms import Independent, Variable, to_data, to_funsor

# Conversion functions use fixed names for Pyro batch dims, but
# accept an event_inputs tuple for custom event dim names.
DIM_TO_NAME = tuple(map("_pyro_dim_{}".format, range(-100, 0)))
NAME_TO_DIM = dict(zip(DIM_TO_NAME, range(-100, 0)))


def default_dim_to_name(inputs_shape, event_inputs):
    dim_to_name_list = DIM_TO_NAME + event_inputs if event_inputs else DIM_TO_NAME
    return OrderedDict(
        zip(
            range(-len(inputs_shape), 0),
            dim_to_name_list[len(dim_to_name_list) - len(inputs_shape) :],
        )
    )


def default_name_to_dim(event_inputs):
    if not event_inputs:
        return NAME_TO_DIM
    dim_to_name = DIM_TO_NAME + event_inputs
    return dict(zip(dim_to_name, range(-len(dim_to_name), 0)))


def tensor_to_funsor(tensor, event_inputs=(), event_output=0, dtype="real"):
    """
    Convert a :class:`torch.Tensor` to a :class:`funsor.tensor.Tensor` .

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
    :rtype: funsor.tensor.Tensor
    """
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(event_inputs, tuple)
    assert isinstance(event_output, int) and event_output >= 0
    inputs_shape = tensor.shape[: tensor.dim() - event_output]
    output = Array[dtype, tensor.shape[tensor.dim() - event_output :]]
    dim_to_name = default_dim_to_name(inputs_shape, event_inputs)
    return to_funsor(tensor, output, dim_to_name)


def funsor_to_tensor(funsor_, ndims, event_inputs=()):
    """
    Convert a :class:`funsor.tensor.Tensor` to a :class:`torch.Tensor` .

    Note this should not touch data, but may trigger a
    :meth:`torch.Tensor.reshape` op.

    :param funsor.tensor.Tensor funsor_: A funsor.
    :param int ndims: The number of result dims, ``== result.dim()``.
    :param tuple event_inputs: Names assigned to rightmost dimensions.
    :return: A PyTorch tensor.
    :rtype: torch.Tensor
    """
    assert isinstance(funsor_, Tensor)
    assert all(k.startswith("_pyro_dim_") or k in event_inputs for k in funsor_.inputs)

    tensor = to_data(funsor_, default_name_to_dim(event_inputs))

    if ndims != tensor.dim():
        tensor = tensor.reshape((1,) * (ndims - tensor.dim()) + tensor.shape)
    assert tensor.dim() == ndims
    return tensor


def dist_to_funsor(pyro_dist, event_inputs=()):
    """
    Convert a PyTorch distribution to a Funsor.

    :param torch.distribution.Distribution: A PyTorch distribution.
    :return: A funsor.
    :rtype: funsor.terms.Funsor
    """
    assert isinstance(pyro_dist, torch.distributions.Distribution)
    assert isinstance(event_inputs, tuple)
    return to_funsor(
        pyro_dist, Real, default_dim_to_name(pyro_dist.batch_shape, event_inputs)
    )


def mvn_to_funsor(pyro_dist, event_inputs=(), real_inputs=OrderedDict()):
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
    :param tuple event_inputs: A tuple of names for rightmost dimensions.
        These will be assigned to ``result.inputs`` of type ``Bint``.
    :param OrderedDict real_inputs: A dict mapping real variable name
        to appropriately sized ``Real``. The sum of all ``.numel()``
        of all real inputs should be equal to the ``pyro_dist`` dimension.
    :return: A funsor with given ``real_inputs`` and possibly additional
        Bint inputs.
    :rtype: funsor.terms.Funsor
    """
    assert isinstance(pyro_dist, torch.distributions.MultivariateNormal)
    assert isinstance(event_inputs, tuple)
    assert isinstance(real_inputs, OrderedDict)
    dim_to_name = default_dim_to_name(pyro_dist.batch_shape, event_inputs)
    funsor_dist = to_funsor(pyro_dist, Real, dim_to_name)
    if len(real_inputs) == 0:
        return funsor_dist
    discrete, gaussian = funsor_dist(value="value").terms
    inputs = OrderedDict(
        (k, v) for k, v in gaussian.inputs.items() if v.dtype != "real"
    )
    inputs.update(real_inputs)
    return discrete + Gaussian(
        white_vec=gaussian.white_vec, prec_sqrt=gaussian.prec_sqrt, inputs=inputs
    )


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
    if isinstance(gaussian, Contraction):
        gaussian = [v for v in gaussian.terms if isinstance(v, Gaussian)][0]
    assert isinstance(gaussian, Gaussian)
    result = to_data(gaussian, name_to_dim=default_name_to_dim(event_inputs))
    if ndims != len(result.batch_shape):
        result = result.expand(
            (1,) * (ndims - len(result.batch_shape)) + result.batch_shape
        )
    return result


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
    assert isinstance(funsor_, Contraction), funsor_
    assert sum(1 for d in funsor_.inputs.values() if d.dtype == "real") == 1
    assert event_inputs, "no components name found"
    assert not any(isinstance(v, Delta) for v in funsor_.terms)
    cat, mvn = to_data(funsor_, name_to_dim=default_name_to_dim(event_inputs))
    if ndims != len(cat.batch_shape):
        cat = cat.expand((1,) * (ndims - len(cat.batch_shape)) + cat.batch_shape)
    if ndims + 1 != len(mvn.batch_shape):
        mvn = mvn.expand((1,) * (ndims + 1 - len(mvn.batch_shape)) + mvn.batch_shape)
    return cat, mvn


def matrix_and_mvn_to_funsor(
    matrix, mvn, event_dims=(), x_name="value_x", y_name="value_y"
):
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
        These will be assigned to ``result.inputs`` of type ``Bint``.
    :param str x_name: The name of the ``x`` random variable.
    :param str y_name: The name of the ``y`` random variable.
    :return: A funsor with given ``real_inputs`` and possibly additional
        Bint inputs.
    :rtype: funsor.terms.Funsor
    """
    assert isinstance(mvn, torch.distributions.MultivariateNormal) or (
        isinstance(mvn, torch.distributions.Independent)
        and isinstance(mvn.base_dist, torch.distributions.Normal)
    )
    assert isinstance(matrix, torch.Tensor)
    x_size, y_size = matrix.shape[-2:]
    assert mvn.event_shape == (y_size,)

    # Handle diagonal normal distributions as an efficient special case.
    if isinstance(mvn, torch.distributions.Independent):
        # Create an i-batched Gaussian over x and y_i.
        log_prob = -0.5 * y_size * math.log(
            2 * math.pi
        ) - mvn.base_dist.scale.log().sum(-1)
        log_prob = tensor_to_funsor(log_prob, event_dims)

        matrix_x = ops.transpose(matrix, -1, -2)  # [...,Y,X]
        matrix_y = ops.new_full(matrix_x, matrix_x.shape[:-1] + (1,), -1.0)  # [...,Y,1]
        matrix_xy = ops.cat([matrix_x, matrix_y], -1)  # [...,Y,X+1]
        prec_sqrt = (matrix_xy / mvn.base_dist.scale[..., None])[..., None]
        white_vec = (-mvn.base_dist.loc / mvn.base_dist.scale)[..., None]

        i = Variable(gensym("i"), Bint[y_size])
        y_i = Variable(gensym(f"{y_name}_i"), Real)
        inputs = log_prob.inputs.copy()
        inputs[i.name] = i.output
        inputs[x_name] = Reals[x_size]
        inputs[y_i.name] = Real
        g_i = Gaussian(white_vec=white_vec, prec_sqrt=prec_sqrt, inputs=inputs)

        # Convert to a joint Gaussian over x and y, possibly lazily.
        # This expands the y part of the matrix from linear to square,
        # incurring asymptotic increase O((X+1)Y) ==> O((X+Y)Y).
        #
        #   [ ? ? ? ? ? | ? ]      [ ? ? ? ? ? | ? . . . ]
        #   [ ? ? ? ? ? | ? ] ===> [ ? ? ? ? ? | . ? . . ]
        #   [ ? ? ? ? ? | ? ]      [ ? ? ? ? ? | . . ? . ]
        #   [ ? ? ? ? ? | ? ]      [ ? ? ? ? ? | . . . ? ]
        g = Independent(g_i, y_name, i.name, y_i.name)
        # Equivalently, g_i(**{y_i.name: y[i]}).reduce(ops.add, i)
        return g + log_prob

    # Create a rank-y Gaussian over (x,y).
    log_prob = -0.5 * y_size * math.log(2 * math.pi) - mvn.scale_tril.diagonal(
        dim1=-1, dim2=-2
    ).log().sum(-1)
    log_prob = tensor_to_funsor(log_prob, event_dims)

    prec_sqrt_y = ops.transpose(ops.triangular_inv(mvn.scale_tril), -1, -2)
    prec_sqrt_xy = matrix @ prec_sqrt_y
    prec_sqrt_yy = (-prec_sqrt_y).expand(prec_sqrt_xy.shape[:-2] + (-1, -1))
    prec_sqrt = ops.cat([prec_sqrt_xy, prec_sqrt_yy], -2)
    white_vec = (-mvn.loc[..., None, :] @ prec_sqrt_y)[..., 0, :]
    white_vec = white_vec.expand(prec_sqrt.shape[:-2] + (-1,))

    # Note the round trip tensor_to_funsor(...).data strips leading 1's from the shape.
    white_vec = tensor_to_funsor(white_vec, event_dims, 1)
    prec_sqrt = tensor_to_funsor(prec_sqrt, event_dims, 2)
    inputs = white_vec.inputs.copy()
    inputs[x_name] = Reals[x_size]
    inputs[y_name] = Reals[y_size]
    g = Gaussian(white_vec=white_vec.data, prec_sqrt=prec_sqrt.data, inputs=inputs)
    return g + log_prob
