# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import inspect
import typing
import warnings
from collections import OrderedDict
from functools import singledispatch

import makefun

from funsor.instrument import debug_logged
from funsor.terms import (
    Funsor,
    FunsorMeta,
    Subs,
    Variable,
    eager,
    substitute,
    to_funsor,
)
from funsor.util import as_callable


def _get_name(fn):
    return getattr(fn, "__name__", type(fn).__name__)


def _erase_types(fn):
    def result(*args):
        return fn(*args)

    result.__name__ = _get_name(fn)
    result.__module__ = fn.__module__
    return debug_logged(result)


class FreshMeta(type):
    def __getitem__(cls, fn):
        return Fresh(fn)


class Fresh(metaclass=FreshMeta):
    """
    Type hint for :func:`make_funsor` decorated functions. This provides hints
    for fresh variables (names) and the return type.

    Examples::

        Fresh[Real]  # a constant known domain
        Fresh[lambda x: Array[x.dtype, x.shape[1:]]  # args are Domains
        Fresh[lambda x, y: Bint[x.size + y.size]]

    :param callable fn: A lambda taking named arguments (in any order)
        which will be filled in with the domain of the similarly named
        funsor argument to the decorated function. This lambda should
        compute a desired resulting domain given domains of arguments.
    """

    def __init__(self, fn):
        function = type(lambda: None)
        self.fn = fn if isinstance(fn, function) else lambda: fn
        self.args = inspect.getfullargspec(fn)[0]

    def __call__(self, **kwargs):
        return self.fn(*map(kwargs.__getitem__, self.args))


class Bound:
    """
    Type hint for :func:`make_funsor` decorated functions. This provides hints
    for bound variables (names).
    """

    pass


class ValueMeta(type):
    def __getitem__(cls, value_type):
        return Value(value_type)


class Value(metaclass=ValueMeta):
    def __init__(self, value_type):
        if issubclass(value_type, Funsor):
            raise TypeError("Types cannot depend on Funsor values")
        self.value_type = value_type


class HasMeta(type):
    def __getitem__(cls, bound):
        return Has(bound)


class Has(metaclass=HasMeta):
    """
    Type hint for :func:`make_funsor` decorated functions.

    This hint asserts that a set of :class:`Bound` variables
    always appear in the ``.inputs`` of the annotated argument.

    For example, we could write a named ``matmul`` function that
    asserts that both arguments always contain the reduced input,
    and cannot be constant with respect to that input::

        @make_funsor
        def MatMul(
            x: Has[{"i"}],
            y: Has[{"i"}],
            i: Bound,
        ) -> Fresh[lambda x: x]:
            return (x * y).reduce(ops.add, i)

    Here the string ``"i"`` in the annotations for ``x`` and ``y``
    refer to the argument ``i`` of our ``MatMul`` function,
    which is known to be ``Bound`` (i.e it does not appear in the
    ``.inputs`` of evaluating ``Matmul(x, y, "i")``.

    .. warning ::

        This annotation is experimental and may be removed in the future.

        Note that because Funsor is inherently extensional,
        violating a `Has` constraint only raises a :class:`SyntaxWarning`
        rather than a full :class:`TypeError`  and even then only under
        the :func:`~funsor.interpretations.reflect`  interpretation.

        As such, :class:`Has` annotations should be used sparingly,
        reserved for cases where the programmer has complete control
        over the inputs to a function and knows that an argument
        will always depend on a bound variable, e.g. when writing one-off
        Funsor terms to describe custom layers in a neural network.

    :param set bound: A :class:`~builtins.set` of strings of names of
        :class:`Bound` arguments of a :func:`make_funsor` -decorated function.
    """

    def __init__(self, bound):
        assert isinstance(bound, set)
        assert all(isinstance(v, str) for v in bound)
        self.bound = bound


def _get_dependent_args(fields, hints, args):
    return {
        name: arg if isinstance(hint, Value) else arg.output
        for name, arg, hint in zip(fields, args, hints)
        if hint in (Funsor, Bound)
        or isinstance(hint, (Has, Value))
        or (isinstance(hint, Fresh) and name in hint.args)
    }


def make_funsor(fn):
    """
    Decorator to dynamically create a subclass of
    :class:`~funsor.terms.Funsor`, together with a single default eager
    pattern.

    This infers inputs, outputs, fresh, and bound variables from type hints
    follow the following convention:

    - Funsor inputs are typed :class:`~funsor.terms.Funsor`.
    - Bound variable inputs (names) are typed :class:`Bound`.
    - Fresh variable inputs (names) are typed :class:`Fresh` together with
      lambda to compute the dependent domain.
    - Ground value inputs (e.g. Python ints) are typed :class:`Value` together with
      their actual data type, e.g. ``Value[int]``.
    - The return value is typed :class:`Fresh` together with a lambda to
      compute the dependent return domain.

    For example to unflatten a single coordinate into a pair of coordinates we
    could define::

        @make_funsor
        def Unflatten(
            x: Funsor,
            i: Bound,
            i_over_2: Fresh[lambda i: Bint[i.size // 2]],
            i_mod_2: Fresh[lambda: Bint[2]],
        ) -> Fresh[lambda x: x]:
            assert i.output.size % 2 == 0
            return x(**{i.name: i_over_2 * Number(2, 3) + i_mod_2})

    :param callable fn: A type annotated function of Funsors.
    :rtype: subclas of :class:`~funsor.terms.Funsor`
    """
    input_types = typing.get_type_hints(as_callable(fn))
    for name, hint in input_types.items():
        if not (hint in (Funsor, Bound) or isinstance(hint, (Fresh, Value, Has))):
            raise TypeError(f"Invalid type hint {name}: {hint}")
    if any(
        isinstance(hint, Fresh) and arg in hint.args
        for arg, hint in input_types.items()
    ):
        input_types["bind_return"] = Value[frozenset]

        def new_fn(*args):
            args, bind_return = args[:-1], args[-1]
            result = fn(*args)
            return Subs(result, bind_return)

    else:
        new_fn = fn

    output_type = input_types.pop("return")
    hints = tuple(input_types.values())

    class ResultMeta(FunsorMeta):
        def __call__(cls, *args, bind_return=None):
            args = list(args)

            # Bind-and-return variables
            if bind_return is None:
                bind_return = frozenset(
                    (arg, arg)
                    for hint, arg, arg_name in zip(hints, args, cls._ast_fields)
                    if isinstance(hint, Fresh) and arg_name in hint.args
                )

            # Compute domains of bound variables.
            for i, (name, arg) in enumerate(zip(cls._ast_fields, args)):
                hint = input_types[name]
                if hint is Funsor or isinstance(hint, Has):  # TODO support domains
                    args[i] = to_funsor(arg)
                elif hint is Bound or (isinstance(hint, Fresh) and name in hint.args):
                    for other in args:
                        if isinstance(other, Funsor):
                            domain = other.inputs.get(arg, None)
                            if domain is not None:
                                arg = to_funsor(arg, domain)
                    if not isinstance(arg, Variable):
                        raise ValueError(f"Cannot infer domain of {name}={arg}")
                    args[i] = arg
                elif isinstance(hint, Value):
                    if not isinstance(arg, hint.value_type):
                        raise TypeError(
                            f"invalid dependent value type: {arg}: {hint.value_type}"
                        )
                    args[i] = arg

            # Compute domains of fresh variables.
            dependent_args = _get_dependent_args(cls._ast_fields, hints, args)
            for i, (hint, arg, arg_name) in enumerate(
                zip(hints, args, cls._ast_fields)
            ):
                if isinstance(hint, Fresh) and arg_name in hint.args:
                    domain = hint(**dependent_args)
                    args[i] = to_funsor(arg.name, domain)
                elif isinstance(hint, Fresh):
                    domain = hint(**dependent_args)
                    args[i] = to_funsor(arg, domain)

            # Append bind_return to args
            if bind_return:
                args.append(bind_return)
            return super().__call__(*args)

    @makefun.with_signature(
        "__init__({})".format(", ".join(["self"] + list(input_types)))
    )
    def __init__(self, **kwargs):
        args = tuple(kwargs[k] for k in self._ast_fields)
        bind_return = dict(kwargs.get("bind_return", dict()))
        dependent_args = _get_dependent_args(self._ast_fields, hints, args)
        output = output_type(**dependent_args)
        inputs = OrderedDict()
        bound = {}
        fresh = frozenset()
        for hint, arg, arg_name in zip(hints, args, self._ast_fields):
            if hint is Funsor:
                assert isinstance(arg, Funsor)
                inputs.update(arg.inputs)
            elif isinstance(hint, Has):
                assert isinstance(arg, Funsor)
                inputs.update(arg.inputs)
                for name in hint.bound:
                    if kwargs[name].name not in arg.inputs:
                        warnings.warn(
                            f"Argument {arg_name} is missing bound variable {kwargs[name]} from argument {name}."
                            f"Are you sure {name} will always appear in {arg_name}?",
                            SyntaxWarning,
                        )
        for hint, arg, arg_name in zip(hints, args, self._ast_fields):
            if hint is Bound:
                bound[arg.name] = inputs.pop(arg.name)
            elif isinstance(hint, Fresh) and arg_name in hint.args:
                bound[arg.name] = inputs.pop(arg.name)
                inputs[bind_return[arg.name]] = arg.output
                fresh |= frozenset({bind_return[arg.name]})
        for hint, arg in zip(hints, args):
            if isinstance(hint, Fresh):
                if arg.name not in bound:
                    inputs[arg.name] = arg.output
                    fresh |= frozenset({arg.name})
        Funsor.__init__(self, inputs, output, fresh, bound)
        for name, arg in zip(self._ast_fields, args):
            if name == "bind_return":
                arg = dict(arg)
            setattr(self, name, arg)

    def _alpha_convert(self, alpha_subs):
        result = []
        new_alpha_subs = {k: to_funsor(v, self.bound[k]) for k, v in alpha_subs.items()}
        for hint, value, arg_name in zip(hints, self._ast_values, self._ast_fields):
            if isinstance(hint, Fresh) and arg_name in hint.args:
                result.append(to_funsor(alpha_subs[value.name], value.output))
            elif arg_name == "bind_return":
                result.append(
                    frozenset(
                        (alpha_subs.get(k, k), v) for k, v in self.bind_return.items()
                    )
                )
            else:
                result.append(substitute(value, new_alpha_subs))
        return tuple(result)

    name = _get_name(fn)
    ResultMeta.__name__ = f"{name}Meta"
    Result = ResultMeta(
        name, (Funsor,), {"__init__": __init__, "_alpha_convert": _alpha_convert}
    )
    pattern = (Result,) + tuple(
        _hint_to_pattern(input_types[k]) for k in Result._ast_fields
    )
    eager.register(*pattern)(_erase_types(new_fn))
    return Result


@singledispatch
def _hint_to_pattern(t):
    return Funsor


@_hint_to_pattern.register(Value)
def _(t):
    return t.value_type
