# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Ldentifier: Apache-2.0

from collections import OrderedDict

import pytest

from funsor.domains import Array, Bint, Dependent, Product, Reals
from funsor.op_factory import make_op
from funsor.tensor import Tensor
from funsor.terms import Binary, Tuple, Unary, Variable
from funsor.testing import assert_close, randn, requires_backend


@pytest.mark.parametrize("batch_shape", [(), (10,)], ids=str)
def test_make_op_1(batch_shape):
    inputs = OrderedDict((k, Bint[s]) for k, s in zip("abc", batch_shape))

    weights = randn(28 * 28, 2 * 20)
    bias = randn(2 * 20)

    @make_op
    def encode(image: Reals[28, 28]) -> Product[Reals[20], Reals[20]]:
        batch_shape = image.shape[:-2]
        loc_scale = image.reshape(*batch_shape, 28 * 28) @ weights + bias
        loc_scale = loc_scale.reshape(*batch_shape, 2, 20)
        loc = loc_scale[..., 0, :]
        scale = loc_scale[..., 1, :]
        assert loc.shape == batch_shape + (20,)
        assert scale.shape == batch_shape + (20,)
        return loc, scale

    # Check symbolic.
    image = Variable("image", Reals[28, 28])
    actual = encode(image)
    assert isinstance(actual, Unary)
    assert actual.output == Product[Reals[20], Reals[20]]

    # Check on ground term.
    image_data = randn(*batch_shape, 28, 28)
    actual_data = encode(image_data)
    assert isinstance(actual_data, tuple)
    assert len(actual_data) == 2
    assert isinstance(actual_data[0], type(image_data))
    assert isinstance(actual_data[1], type(image_data))

    # Check on funsor.Tensors.
    image = Tensor(image_data, inputs)
    actual = encode(image)
    assert isinstance(actual, Tuple)
    assert len(actual) == 2
    assert isinstance(actual[0], Tensor)
    assert isinstance(actual[1], Tensor)
    assert_close(actual[0].data, actual_data[0])
    assert_close(actual[1].data, actual_data[1])


@pytest.mark.parametrize("batch_shape", [(), (10,)], ids=str)
def test_make_op_2(batch_shape):
    inputs = OrderedDict((k, Bint[s]) for k, s in zip("abc", batch_shape))

    @make_op
    def matmul(x: Reals[2, 3], y: Reals[3, 4]) -> Reals[2, 4]:
        return x @ y

    # Check symbolic.
    x = Variable("x", Reals[2, 3])
    y = Variable("y", Reals[3, 4])
    actual = matmul(x, y)
    assert isinstance(actual, Binary)
    assert actual.output == Reals[2, 4]

    # Check on ground term.
    x_data = randn(*batch_shape, 2, 3)
    y_data = randn(*batch_shape, 3, 4)
    actual_data = matmul(x_data, y_data)
    assert isinstance(actual_data, type(x_data))
    assert_close(actual_data, x_data @ y_data)

    # Check on funsor.Tensors.
    x = Tensor(x_data, inputs)
    y = Tensor(y_data, inputs)
    actual = matmul(x, y)
    assert isinstance(actual, Tensor)
    assert_close(actual, Tensor(actual_data, inputs))


@pytest.mark.parametrize("batch_shape", [(), (10,)], ids=str)
def test_make_op_3(batch_shape):
    inputs = OrderedDict((k, Bint[s]) for k, s in zip("abc", batch_shape))

    @make_op
    def matmul(
        x: Array,
        y: Array,
    ) -> Dependent[lambda x, y: Reals[x.shape[0], y.shape[1]]]:
        return x @ y

    for L, J, K in [(2, 3, 4), (5, 6, 7)]:
        # Check symbolic.
        x = Variable("x", Reals[L, J])
        y = Variable("y", Reals[J, K])
        actual = matmul(x, y)
        assert isinstance(actual, Binary)
        assert actual.output == Reals[L, K]

        # Check on ground term.
        x_data = randn(*batch_shape, L, J)
        y_data = randn(*batch_shape, J, K)
        actual_data = matmul(x_data, y_data)
        assert isinstance(actual_data, type(x_data))
        assert_close(actual_data, x_data @ y_data)

        # Check on funsor.Tensors.
        x = Tensor(x_data, inputs)
        y = Tensor(y_data, inputs)
        actual = matmul(x, y)
        assert isinstance(actual, Tensor)
        assert_close(actual, Tensor(actual_data, inputs))


@requires_backend("torch", reason="requires nn.Module")
@pytest.mark.parametrize("batch_shape", [(), (10,)], ids=str)
def test_nn_module(batch_shape):
    import torch

    class Matmul(torch.nn.Module):
        def forward(
            self, x, y
        ) -> Dependent[lambda x, y: Reals[x.shape[0], y.shape[1]]]:
            return x @ y

    matmul = make_op(Matmul())

    inputs = OrderedDict((k, Bint[s]) for k, s in zip("abc", batch_shape))
    for L, J, K in [(2, 3, 4), (5, 6, 7)]:
        # Check symbolic.
        x = Variable("x", Reals[L, J])
        y = Variable("y", Reals[J, K])
        actual = matmul(x, y)
        assert isinstance(actual, Binary)
        assert actual.output == Reals[L, K]

        # Check on ground term.
        x_data = randn(*batch_shape, L, J)
        y_data = randn(*batch_shape, J, K)
        actual_data = matmul(x_data, y_data)
        assert isinstance(actual_data, type(x_data))
        assert_close(actual_data, x_data @ y_data)

        # Check on funsor.Tensors.
        x = Tensor(x_data, inputs)
        y = Tensor(y_data, inputs)
        actual = matmul(x, y)
        assert isinstance(actual, Tensor)
        assert_close(actual, Tensor(actual_data, inputs))
