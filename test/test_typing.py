# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, FrozenSet, Optional, Tuple, Union

import pytest
from multipledispatch import dispatch

from funsor.ops import AssociativeOp, Op
from funsor.registry import PartialDispatcher
from funsor.terms import Cat, Funsor, Number, Reduce, Stack, Variable
from funsor.typing import (
    GenericTypeMeta,
    Variadic,
    deep_isinstance,
    deep_issubclass,
    deep_type,
    get_args,
    get_origin,
    get_type_hints,
    typing_wrap,
)


def test_deep_issubclass_generic_identity():
    assert deep_issubclass(Reduce, Reduce)
    assert deep_issubclass(
        Reduce[AssociativeOp, Funsor, frozenset],
        Reduce[AssociativeOp, Funsor, frozenset],
    )
    assert deep_issubclass(Tuple, Tuple)


def test_deep_issubclass_generic_empty():
    assert deep_issubclass(Reduce[AssociativeOp, Funsor, frozenset], Funsor)
    assert deep_issubclass(Reduce[AssociativeOp, Funsor, frozenset], Reduce)
    assert not deep_issubclass(Funsor, Reduce[AssociativeOp, Funsor, frozenset])
    assert not deep_issubclass(Reduce, Reduce[AssociativeOp, Funsor, frozenset])


def test_deep_issubclass_generic_neither():
    assert not deep_issubclass(
        Reduce[AssociativeOp, Reduce[AssociativeOp, Funsor, frozenset], frozenset],
        Reduce[Op, Variable, frozenset],
    )
    assert not deep_issubclass(
        Reduce[Op, Variable, frozenset],
        Reduce[AssociativeOp, Reduce[AssociativeOp, Funsor, frozenset], frozenset],
    )

    assert not deep_issubclass(
        Stack[str, Tuple[Number, Number]], Stack[str, Tuple[Number, Reduce]]
    )
    assert not deep_issubclass(
        Stack[str, Tuple[Number, Reduce]], Stack[str, Tuple[Number, Number]]
    )


def test_deep_issubclass_generic_tuple_internal():
    assert deep_issubclass(Stack[str, Tuple[Number, Number, Number]], Stack)
    assert deep_issubclass(Stack[str, Tuple[Number, Number, Number]], Stack[str, tuple])
    assert not deep_issubclass(Stack, Stack[str, Tuple[Number, Number, Number]])
    assert not deep_issubclass(
        Stack[str, tuple], Stack[str, Tuple[Number, Number, Number]]
    )
    assert not deep_issubclass(
        Stack[str, Tuple[Number, Number]], Stack[str, Tuple[Number, Reduce]]
    )


def test_deep_issubclass_generic_union_internal():

    assert deep_issubclass(
        Reduce[AssociativeOp, Union[Number, Funsor], frozenset], Funsor
    )
    assert not deep_issubclass(
        Funsor, Reduce[AssociativeOp, Union[Number, Funsor], frozenset]
    )

    assert deep_issubclass(
        Reduce[
            AssociativeOp,
            Union[Number, Stack[str, Tuple[Number, Number]]],
            frozenset,
        ],
        Funsor,
    )
    assert deep_issubclass(
        Reduce[AssociativeOp, Union[Number, Stack], frozenset],
        Reduce[Op, Funsor, frozenset],
    )

    assert deep_issubclass(
        Reduce[AssociativeOp, Union[Number, Stack], frozenset],
        Reduce[AssociativeOp, Funsor, frozenset],
    )
    assert not deep_issubclass(
        Reduce[AssociativeOp, Funsor, frozenset],
        Reduce[AssociativeOp, Union[Number, Stack], frozenset],
    )
    assert not deep_issubclass(
        Reduce[Op, Funsor, frozenset],
        Reduce[AssociativeOp, Union[Number, Stack], frozenset],
    )


def test_deep_issubclass_generic_union_internal_multiple():
    assert not deep_issubclass(
        Reduce[Union[Op, AssociativeOp], Stack, frozenset],
        Reduce[
            AssociativeOp,
            Union[
                Stack[str, tuple],
                Reduce[AssociativeOp, Union[Cat, Stack], frozenset],
            ],
            frozenset,
        ],
    )

    assert not deep_issubclass(
        Reduce[
            AssociativeOp,
            Union[Stack, Reduce[AssociativeOp, Union[Number, Stack], frozenset]],
            frozenset,
        ],
        Reduce[Union[Op, AssociativeOp], Stack, frozenset],
    )


def test_deep_issubclass_tuple_variadic():

    assert deep_issubclass(Tuple[int, ...], Tuple)
    assert deep_issubclass(Tuple[int], Tuple[int, ...])
    assert deep_issubclass(Tuple[int, int], Tuple[int, ...])

    assert not deep_issubclass(Tuple[int, ...], Tuple[int])
    assert not deep_issubclass(Tuple[int, ...], Tuple[int, int])

    assert deep_issubclass(Tuple[Reduce, ...], Tuple[Funsor, ...])
    assert not deep_issubclass(Tuple[Funsor, ...], Tuple[Reduce, ...])

    assert deep_issubclass(Tuple[Reduce], Tuple[Funsor, ...])
    assert not deep_issubclass(Tuple[Funsor], Tuple[Reduce, ...])

    assert not deep_issubclass(Tuple[str], Tuple[int, ...])
    assert not deep_issubclass(Tuple[int, str], Tuple[int, ...])

    assert deep_issubclass(Tuple[Tuple[int, str]], Tuple[Tuple, ...])
    assert deep_issubclass(Tuple[Tuple[int, str], Tuple[int, str]], Tuple[Tuple, ...])
    assert deep_issubclass(Tuple[Tuple[int, str]], Tuple[Tuple[int, str], ...])
    assert deep_issubclass(Tuple[Tuple[int, str], ...], Tuple[Tuple[int, str], ...])
    assert deep_issubclass(
        Tuple[Tuple[int, str], Tuple[int, str]], Tuple[Tuple[int, str], ...]
    )


def test_deep_type_tuple():

    x1 = (1, 1.5, "a")
    expected_type1 = Tuple[int, float, str]
    assert deep_type(x1) is expected_type1
    assert deep_isinstance(x1, expected_type1)

    x2 = (1, (2, 3))
    expected_type2 = Tuple[int, Tuple[int, int]]
    assert deep_type(x2) is expected_type2
    assert deep_isinstance(x2, expected_type2)


def test_deep_type_frozenset():

    x1 = frozenset(["a", "b"])
    expected_type1 = FrozenSet[str]
    assert deep_type(x1) is expected_type1
    assert deep_isinstance(x1, expected_type1)

    with pytest.raises(NotImplementedError):
        x2 = frozenset(["a", 1])
        deep_type(x2)


def test_generic_type_cons_hash():
    class A(metaclass=GenericTypeMeta):
        pass

    class B(metaclass=GenericTypeMeta):
        pass

    assert A[int] is A[int]
    assert A[float] is not A[int]
    assert B[int] is not A[int]
    assert B[A[int], int] is B[A[int], int]

    assert FrozenSet[int] is FrozenSet[int]
    assert FrozenSet[B[int]] is FrozenSet[B[int]]

    assert Tuple[B[int, int], ...] is Tuple[B[int, int], ...]

    assert Union[B[int]] is Union[B[int]]
    assert Union[B[int], B[int]] is Union[B[int]]


def test_get_origin():

    assert get_origin(Any) is Any

    assert get_origin(Tuple[int]) in (tuple, Tuple)
    assert get_origin(FrozenSet[int]) in (frozenset, FrozenSet)

    assert get_origin(Union[int, int, str]) is Union

    assert get_origin(Reduce[AssociativeOp, Funsor, frozenset]) is Reduce
    assert get_origin(Reduce) is Reduce


def test_get_args():
    assert not get_args(Any)

    assert get_args(Tuple[int]) == (int,)
    assert get_args(Tuple[int, ...]) == (int, ...)
    assert not get_args(Tuple)

    assert get_args(FrozenSet[int]) == (int,)

    assert int in get_args(Optional[int])

    assert get_args(Union[int]) == ()
    assert get_args(Union[int, str]) == (int, str)
    assert get_args(Union[int, int, str]) == (int, str)

    assert get_args(Reduce[AssociativeOp, Funsor, frozenset]) == (
        AssociativeOp,
        Funsor,
        frozenset,
    )
    assert not get_args(Reduce)


def test_get_type_hints():
    def f(a: Tuple[int, ...], b: Reduce[AssociativeOp, Funsor, frozenset]) -> int:
        return 0

    hints = get_type_hints(f)
    assert hints == {
        "a": Tuple[int, ...],
        "b": Reduce[AssociativeOp, Funsor, frozenset],
        "return": int,
    }

    hints.pop("return")
    assert "return" in get_type_hints(f)


def test_variadic_dispatch_basic():
    @dispatch(Variadic[object])
    def f(*args):
        return 1

    @dispatch(int, int)
    def f(a, b):
        return 2

    @dispatch(Variadic[int])
    def f(*args):
        return 3

    @dispatch(typing_wrap(Tuple), typing_wrap(Tuple))
    def f(a, b):
        return 4

    @dispatch(Variadic[Tuple])
    def f(*args):
        return 5

    assert f(1.5) == 1
    assert f(1.5, 1) == 1

    assert f(1, 1) == 2
    assert f(1) == 3
    assert f(1, 2, 3) == 3

    assert f((1, 1), (1, 1)) == 4
    assert f((1, 2)) == 5
    assert f((1, 2), (3, 4), (5, 6)) == 5


def test_dispatch_typing():

    f = PartialDispatcher(lambda *args: 1)

    @f.register()
    def f2(a: int, b: int) -> int:
        return 2

    @f.register()
    def f3(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return 3

    @f.register()
    def f4(a: Tuple[int, ...], b: Tuple[int, int]) -> int:
        return 4

    @f.register()
    def f5(a: Tuple[int, float], b: Tuple[int, float]) -> int:
        return 5

    assert f(1.5) == 1
    assert f(1.5, 1) == 1

    assert f(1, 1) == 2

    assert f((1, 1), (1, 1)) == 3
    assert f((1, 2)) == 1
    assert f((1, 2, 3), (4, 5)) == 4

    assert f((1, 1.5), (2, 2.5)) == 5


def test_variadic_dispatch_typing():

    f = PartialDispatcher(lambda *args: 1)

    @f.register()
    def _(a: int, b: int) -> int:
        return 2

    @f.register([int])
    def _(*args):
        return 3

    @f.register()
    def _(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return 4

    @f.register([Tuple[int, int]])  # list syntax for variadic
    def _(*args):
        return 5

    @f.register()
    def _(a: Tuple[int, float], b: Tuple[int, float]) -> int:
        return 6

    assert f(1.5) == 1
    assert f(1.5, 1) == 1

    assert f(1, 1) == 2
    assert f(1) == 3
    assert f(1, 2, 3) == 3

    assert f((1, 1), (1, 1)) == 4
    assert f((1, 2)) == 5
    assert f((1, 2), (3, 4), (5, 6)) == 5

    assert f((1, 1.5), (2, 2.5)) == 6
