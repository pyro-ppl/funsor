# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import inspect
import typing
from collections import OrderedDict

import makefun

from funsor.interpreter import debug_logged
from funsor.terms import Funsor, FunsorMeta, Variable, eager, to_funsor


def _erase_types(fn):
    def result(*args):
        return fn(*args)
    result.__name__ = fn.__name__
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
        self.args = inspect.getargspec(fn)[0]

    def __call__(self, **kwargs):
        return self.fn(*map(kwargs.__getitem__, self.args))


class Bound:
    """
    Type hint for :func:`make_funsor` decorated functions. This provides hints
    for bound variables (names).
    """
    pass


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
    input_types = typing.get_type_hints(fn)
    for name, hint in input_types.items():
        if not (hint in (Funsor, Bound) or isinstance(hint, Fresh)):
            raise TypeError(f"Invalid type hint {name}: {hint}")
    output_type = input_types.pop("return")
    hints = tuple(input_types.values())

    class ResultMeta(FunsorMeta):
        def __call__(cls, *args):
            args = list(args)

            # Compute domains of bound variables.
            for i, (name, arg) in enumerate(zip(cls._ast_fields, args)):
                hint = input_types[name]
                if hint is Funsor:  # TODO support domains
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

            # Compute domains of fresh variables.
            dependent_args = {name: arg.output
                              for name, arg, hint in zip(cls._ast_fields, args, hints)
                              if hint in (Funsor, Bound)}
            for i, (hint, arg) in enumerate(zip(hints, args)):
                if isinstance(hint, Fresh):
                    domain = hint(**dependent_args)
                    args[i] = to_funsor(arg, domain)
            return super().__call__(*args)

    @makefun.with_signature("__init__({})".format(", ".join(["self"] + list(input_types))))
    def __init__(self, **kwargs):
        args = tuple(kwargs[k] for k in self._ast_fields)
        dependent_args = {name: arg.output
                          for name, arg, hint in zip(self._ast_fields, args, hints)
                          if hint in (Funsor, Bound)}
        output = output_type(**dependent_args)
        inputs = OrderedDict()
        fresh = set()
        bound = {}
        for hint, arg in zip(hints, args):
            if hint is Funsor:
                inputs.update(arg.inputs)
        for hint, arg in zip(hints, args):
            if hint is Bound:
                bound[arg.name] = inputs.pop(arg.name)
        for hint, arg in zip(hints, args):
            if isinstance(hint, Fresh):
                fresh.add(arg.name)
                inputs[arg.name] = arg.output
        fresh = frozenset(fresh)
        Funsor.__init__(self, inputs, output, fresh, bound)
        for name, arg in zip(self._ast_fields, args):
            setattr(self, name, arg)

    ResultMeta.__name__ = f"{fn.__name__}Meta"
    Result = ResultMeta(fn.__name__, (Funsor,), {"__init__": __init__})
    pattern = (Result,) + (Funsor,) * len(input_types)
    eager.register(*pattern)(_erase_types(fn))
    return Result
