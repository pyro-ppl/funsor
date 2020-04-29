# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Like funsor.pyro.convert
"""
# TODO: add docs and rename pyro_foo variables

import math
from collections import OrderedDict

import numpyro.distributions as dist

import funsor.ops as ops
from funsor.cnf import Contraction
from funsor.delta import Delta
from funsor.distributions import Normal
from funsor.domains import Domain, bint, reals
from funsor.gaussian import Gaussian
from funsor.interpreter import gensym
from funsor.tensor import Tensor, align_tensors
from funsor.terms import Funsor, Independent, Variable, eager, to_data, to_funsor
from funsor.util import broadcast_shape

# Conversion functions use fixed names for Pyro batch dims, but
# accept an event_inputs tuple for custom event dim names.
DIM_TO_NAME = tuple(map("_pyro_dim_{}".format, range(-100, 0)))
NAME_TO_DIM = dict(zip(DIM_TO_NAME, range(-100, 0)))


def default_dim_to_name(inputs_shape, event_inputs):
    dim_to_name_list = DIM_TO_NAME + event_inputs if event_inputs else DIM_TO_NAME
    return OrderedDict(zip(
        range(-len(inputs_shape), 0),
        dim_to_name_list[len(dim_to_name_list) - len(inputs_shape):]))


def default_name_to_dim(event_inputs):
    if not event_inputs:
        return NAME_TO_DIM
    dim_to_name = DIM_TO_NAME + event_inputs
    return dict(zip(dim_to_name, range(-len(dim_to_name), 0)))


def tensor_to_funsor(tensor, event_inputs=(), event_output=0, dtype="real"):
    assert ops.is_numeric_array(tensor)
    assert isinstance(event_inputs, tuple)
    assert isinstance(event_output, int) and event_output >= 0
    inputs_shape = tensor.shape[:len(tensor.shape) - event_output]
    output = Domain(dtype=dtype, shape=tensor.shape[len(tensor.shape) - event_output:])
    dim_to_name = default_dim_to_name(inputs_shape, event_inputs)
    return to_funsor(tensor, output, dim_to_name)


def funsor_to_tensor(funsor_, ndims, event_inputs=()):
    assert isinstance(funsor_, Tensor)
    assert all(k.startswith("_pyro_dim_") or k in event_inputs for k in funsor_.inputs)

    tensor = to_data(funsor_, default_name_to_dim(event_inputs))

    if ndims != len(tensor.shape):
        tensor = tensor.reshape((1,) * (ndims - len(tensor.shape)) + tensor.shape)
    assert len(tensor.shape) == ndims
    return tensor


def dist_to_funsor(pyro_dist, event_inputs=()):
    assert isinstance(pyro_dist, dist.Distribution)
    assert isinstance(event_inputs, tuple)
    return to_funsor(pyro_dist, reals(), default_dim_to_name(pyro_dist.batch_shape, event_inputs))


def mvn_to_funsor(pyro_dist, event_inputs=(), real_inputs=OrderedDict()):
    assert isinstance(pyro_dist, dist.MultivariateNormal)
    assert isinstance(event_inputs, tuple)
    assert isinstance(real_inputs, OrderedDict)
    dim_to_name = default_dim_to_name(pyro_dist.batch_shape, event_inputs)
    return to_funsor(pyro_dist, reals(), dim_to_name, real_inputs=real_inputs)


def funsor_to_mvn(gaussian, ndims, event_inputs=()):
    assert sum(1 for d in gaussian.inputs.values() if d.dtype == "real") == 1
    if isinstance(gaussian, Contraction):
        gaussian = [v for v in gaussian.terms if isinstance(v, Gaussian)][0]
    assert isinstance(gaussian, Gaussian)
    result = to_data(gaussian, name_to_dim=default_name_to_dim(event_inputs))
    if ndims != len(result.batch_shape):
        result = result.expand((1,) * (ndims - len(result.batch_shape)) + result.batch_shape)
    return result


def funsor_to_cat_and_mvn(funsor_, ndims, event_inputs):
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


class AffineNormal(Funsor):
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
    loc += value_x @ matrix
    int_inputs, (loc, scale) = align_tensors(loc, scale, expand=True)

    i_name = gensym("i")
    y_name = gensym("y")
    y_i_name = gensym("y_i")
    int_inputs[i_name] = bint(value_y.output.shape[0])
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

    assert value_y.shape[-1] == loc.shape[-1]
    prec_sqrt = matrix / ops.unsqueeze(scale, -2)
    precision = ops.matmul(prec_sqrt, ops.transpose(prec_sqrt, -1, -2))
    delta = (value_y - loc) / scale
    info_vec = ops.matmul(prec_sqrt, ops.unsqueeze(delta, -1)).squeeze(-1)
    log_normalizer = (-0.5 * loc.shape[-1] * math.log(2 * math.pi)
                      - 0.5 * ops.pow(delta, 2).sum(-1) - ops.log(scale).sum(-1))
    precision = ops.expand(precision, info_vec.shape + (-1,))
    log_normalizer = ops.expand(log_normalizer, info_vec.shape[:-1])
    inputs = int_inputs.copy()
    x_name = gensym("x")
    inputs[x_name] = value_x.output
    x_dist = Tensor(log_normalizer, int_inputs) + Gaussian(info_vec, precision, inputs)
    return x_dist(**{x_name: value_x})


def matrix_and_mvn_to_funsor(matrix, mvn, event_dims=(), x_name="value_x", y_name="value_y"):
    assert (isinstance(mvn, dist.MultivariateNormal) or
            (isinstance(mvn, dist.Independent) and
             isinstance(mvn.base_dist, dist.Normal)))
    assert ops.is_numeric_array(matrix)
    x_size, y_size = matrix.shape[-2:]
    assert mvn.event_shape == (y_size,)

    # Handle diagonal normal distributions as an efficient special case.
    if isinstance(mvn, dist.Independent):
        return AffineNormal(tensor_to_funsor(matrix, event_dims, 2),
                            tensor_to_funsor(mvn.base_dist.loc, event_dims, 1),
                            tensor_to_funsor(mvn.base_dist.scale, event_dims, 1),
                            Variable(x_name, reals(x_size)),
                            Variable(y_name, reals(y_size)))

    info_vec = ops.cholesky_solve(ops.unsqueeze(mvn.loc, -1), mvn.scale_tril).squeeze(-1)
    log_prob = (-0.5 * y_size * math.log(2 * math.pi)
                - ops.log(ops.diagonal(mvn.scale_tril, -1, -2)).sum(-1)
                - 0.5 * (info_vec * mvn.loc).sum(-1))

    batch_shape = broadcast_shape(matrix.shape[:-2], mvn.batch_shape)
    P_yy = ops.expand(mvn.precision_matrix, batch_shape + (y_size, y_size))
    neg_P_xy = ops.matmul(matrix, P_yy)
    P_xy = -neg_P_xy
    P_yx = ops.transpose(P_xy, -1, -2)
    P_xx = ops.matmul(neg_P_xy, ops.transpose(matrix, -1, -2))
    precision = ops.cat(-2, ops.cat(-1, P_xx, P_xy), ops.cat(-1, P_yx, P_yy))
    info_y = ops.expand(info_vec, batch_shape + (y_size,))
    info_x = -ops.matmul(matrix, ops.unsqueeze(info_y, -1)).squeeze(-1)
    info_vec = ops.cat(-1, info_x, info_y)

    info_vec = tensor_to_funsor(info_vec, event_dims, 1)
    precision = tensor_to_funsor(precision, event_dims, 2)
    inputs = info_vec.inputs.copy()
    inputs[x_name] = reals(x_size)
    inputs[y_name] = reals(y_size)
    return tensor_to_funsor(log_prob, event_dims) + Gaussian(info_vec.data, precision.data, inputs)
