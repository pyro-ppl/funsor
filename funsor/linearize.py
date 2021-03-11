# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import inspect
import typing
import warnings
from collections import OrderedDict
from functools import singledispatch

import makefun

from funsor.instrument import debug_logged
from funsor.terms import Funsor, FunsorMeta, Variable, eager, to_funsor



def linearize(fn, primals):
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
    breakpoint()
    input_types = typing.get_type_hints(fn)
    for name, hint in input_types.items():
        if not (hint in (Funsor, Bound) or isinstance(hint, (Fresh, Value, Has))):
            raise TypeError(f"Invalid type hint {name}: {hint}")
    output_type = input_types.pop("return")
    hints = tuple(input_types.values())

    class ResultMeta(FunsorMeta):
        def __call__(cls, *args):
            args = list(args)

            # Compute domains of bound variables.
            for i, (name, arg) in enumerate(zip(cls._ast_fields, args)):
                hint = input_types[name]
                if hint is Funsor or isinstance(hint, Has):  # TODO support domains
                    args[i] = to_funsor(arg)
                elif hint is Bound:
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
            for i, (hint, arg) in enumerate(zip(hints, args)):
                if isinstance(hint, Fresh):
                    domain = hint(**dependent_args)
                    args[i] = to_funsor(arg, domain)
            return super().__call__(*args)

    @makefun.with_signature(
        "__init__({})".format(", ".join(["self"] + list(input_types)))
    )
    def __init__(self, **kwargs):
        args = tuple(kwargs[k] for k in self._ast_fields)
        dependent_args = _get_dependent_args(self._ast_fields, hints, args)
        output = output_type(**dependent_args)
        inputs = OrderedDict()
        bound = {}
        for hint, arg, arg_name in zip(hints, args, self._ast_fields):
            if hint is Funsor:
                assert isinstance(arg, Funsor)
                inputs.update(arg.inputs)
            elif isinstance(hint, Has):
                assert isinstance(arg, Funsor)
                inputs.update(arg.inputs)
                for name in hint.bound:
                    if kwargs[name] not in arg.input_vars:
                        warnings.warn(
                            f"Argument {arg_name} is missing bound variable {kwargs[name]} from argument {name}."
                            f"Are you sure {name} will always appear in {arg_name}?",
                            SyntaxWarning,
                        )
        for hint, arg in zip(hints, args):
            if hint is Bound:
                bound[arg.name] = inputs.pop(arg.name)
        for hint, arg in zip(hints, args):
            if isinstance(hint, Fresh):
                for k, d in arg.inputs.items():
                    if k not in bound:
                        inputs[k] = d
        fresh = frozenset()
        Funsor.__init__(self, inputs, output, fresh, bound)
        for name, arg in zip(self._ast_fields, args):
            setattr(self, name, arg)

    def _alpha_convert(self, alpha_subs):
        alpha_subs = {k: to_funsor(v, self.bound[k]) for k, v in alpha_subs.items()}
        return Funsor._alpha_convert(self, alpha_subs)

    ResultMeta.__name__ = f"{fn.__name__}Meta"
    Result = ResultMeta(
        fn.__name__, (Funsor,), {"__init__": __init__, "_alpha_convert": _alpha_convert}
    )
    pattern = (Result,) + tuple(
        _hint_to_pattern(input_types[k]) for k in Result._ast_fields
    )
    eager.register(*pattern)(_erase_types(fn))
    return Result
