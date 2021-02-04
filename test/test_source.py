# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import ast
import operator
import sys

import pytest

from funsor.source import rewrite_ops_as_vars

python_version = tuple(map(int, sys.version.split()[0].split(".")[:2]))


@pytest.mark.skipif(python_version < (3, 9), reason="not supported")
def test_rewrite_ops_as_vars():
    @rewrite_ops_as_vars({"+": "sum_op", "*": "prod_op"})
    def product_rule(sum_op, prod_op, lhs, rhs, d):
        """foo"""
        return d(lhs) * rhs + lhs * d(rhs)

    actual = product_rule

    def product_rule(sum_op, prod_op, lhs, rhs, d):
        """foo"""
        return sum_op(prod_op(d(lhs), rhs), prod_op(lhs, d(rhs)))

    expected = product_rule

    assert actual.__name__ == expected.__name__
    assert actual.__doc__ == expected.__doc__
    assert actual.__module__ == expected.__module__
    assert actual.__code__.co_code == expected.__code__.co_code
    args = (max, operator.add, 1.23, 4.56, lambda x: x ** 2)
    assert actual(*args) == expected(*args)
