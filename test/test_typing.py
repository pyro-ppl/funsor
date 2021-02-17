# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import typing

from funsor.ops import AssociativeOp, Op
from funsor.terms import Cat, Funsor, Number, Reduce, Stack, Variable
from funsor.typing import deep_issubclass


def test_deep_issubclass_identity():
    assert deep_issubclass(Reduce, Reduce)
    assert deep_issubclass(
        Reduce[AssociativeOp, Funsor, frozenset],
        Reduce[AssociativeOp, Funsor, frozenset],
    )


def test_deep_issubclass_empty():
    assert deep_issubclass(Reduce[AssociativeOp, Funsor, frozenset], Funsor)
    assert deep_issubclass(Reduce[AssociativeOp, Funsor, frozenset], Reduce)
    assert not deep_issubclass(Funsor, Reduce[AssociativeOp, Funsor, frozenset])
    assert not deep_issubclass(Reduce, Reduce[AssociativeOp, Funsor, frozenset])


def test_deep_issubclass_neither():
    assert not deep_issubclass(
        Reduce[AssociativeOp, Reduce[AssociativeOp, Funsor, frozenset], frozenset],
        Reduce[Op, Variable, frozenset],
    )
    assert not deep_issubclass(
        Reduce[Op, Variable, frozenset],
        Reduce[AssociativeOp, Reduce[AssociativeOp, Funsor, frozenset], frozenset],
    )

    assert not deep_issubclass(
        Stack[str, typing.Tuple[Number, Number]],
        Stack[str, typing.Tuple[Number, Reduce]],
    )
    assert not deep_issubclass(
        Stack[str, typing.Tuple[Number, Reduce]],
        Stack[str, typing.Tuple[Number, Number]],
    )


def test_deep_issubclass_tuple_internal():
    assert deep_issubclass(Stack[str, typing.Tuple[Number, Number, Number]], Stack)
    assert deep_issubclass(
        Stack[str, typing.Tuple[Number, Number, Number]], Stack[str, tuple]
    )
    assert not deep_issubclass(Stack, Stack[str, typing.Tuple[Number, Number, Number]])
    assert not deep_issubclass(
        Stack[str, tuple], Stack[str, typing.Tuple[Number, Number, Number]]
    )


def test_deep_issubclass_tuple_finite():
    assert not deep_issubclass(
        Stack[str, typing.Tuple[Number, Number]],
        Stack[str, typing.Tuple[Number, Reduce]],
    )


def test_deep_issubclass_union_internal():

    assert deep_issubclass(
        Reduce[AssociativeOp, typing.Union[Number, Funsor], frozenset], Funsor
    )
    assert not deep_issubclass(
        Funsor, Reduce[AssociativeOp, typing.Union[Number, Funsor], frozenset]
    )

    assert deep_issubclass(
        Reduce[
            AssociativeOp,
            typing.Union[Number, Stack[str, typing.Tuple[Number, Number]]],
            frozenset,
        ],
        Funsor,
    )
    assert deep_issubclass(
        Reduce[AssociativeOp, typing.Union[Number, Stack], frozenset],
        Reduce[Op, Funsor, frozenset],
    )

    assert deep_issubclass(
        Reduce[AssociativeOp, typing.Union[Number, Stack], frozenset],
        Reduce[AssociativeOp, Funsor, frozenset],
    )
    assert not deep_issubclass(
        Reduce[AssociativeOp, Funsor, frozenset],
        Reduce[AssociativeOp, typing.Union[Number, Stack], frozenset],
    )
    assert not deep_issubclass(
        Reduce[Op, Funsor, frozenset],
        Reduce[AssociativeOp, typing.Union[Number, Stack], frozenset],
    )


def test_deep_issubclass_union_internal_multiple():
    assert not deep_issubclass(
        Reduce[typing.Union[Op, AssociativeOp], Stack, frozenset],
        Reduce[
            AssociativeOp,
            typing.Union[
                Stack[str, tuple],
                Reduce[AssociativeOp, typing.Union[Cat, Stack], frozenset],
            ],
            frozenset,
        ],
    )

    assert not deep_issubclass(
        Reduce[
            AssociativeOp,
            typing.Union[
                Stack, Reduce[AssociativeOp, typing.Union[Number, Stack], frozenset]
            ],
            frozenset,
        ],
        Reduce[typing.Union[Op, AssociativeOp], Stack, frozenset],
    )
