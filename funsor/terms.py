r"""
Funsor interpretations
----------------------

Funsor provides three basic interpretations.

- ``reflect`` is completely lazy, even with respect to substitution.
- ``lazy`` substitutes eagerly but performs ops lazily.
- ``eager`` does everything eagerly.

"""

from __future__ import absolute_import, division, print_function

import functools
import itertools
import numbers
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, Hashable
from weakref import WeakValueDictionary

from six import add_metaclass, integer_types
from six.moves import reduce

import funsor.interpreter as interpreter
import funsor.ops as ops
from funsor.domains import Domain, bint, find_domain
from funsor.interpreter import interpret
from funsor.ops import AssociativeOp, Op
from funsor.registry import KeyedRegistry
from funsor.six import getargspec, singledispatch


def reflect(cls, *args):
    """
    Construct a funsor, populate ``._ast_values``, and cons hash.
    """
    cache_key = tuple(id(arg) if not isinstance(arg, Hashable) else arg for arg in args)
    if cache_key in cls._cons_cache:
        return cls._cons_cache[cache_key]
    result = super(FunsorMeta, cls).__call__(*args)
    result._ast_values = args
    cls._cons_cache[cache_key] = result
    return result


_lazy = KeyedRegistry(default=lambda *args: None)
_eager = KeyedRegistry(default=lambda *args: None)


def lazy(cls, *args):
    result = _lazy(cls, *args)
    if result is None:
        result = reflect(cls, *args)
    return result


def eager(cls, *args):
    result = _eager(cls, *args)
    if result is None:
        result = reflect(cls, *args)
    return result


lazy.register = _lazy.register
eager.register = _eager.register

interpreter.set_interpretation(eager)  # Use eager interpretation by default.


class FunsorMeta(ABCMeta):
    """
    Metaclass for Funsors to perform three independent tasks:

    1.  Fill in default kwargs and convert kwargs to args before deferring to a
        nonstandard interpretation. This allows derived metaclasses to fill in
        defaults and do type conversion, thereby simplifying logic of
        interpretations.
    2.  Ensure each Funsor class has an attribute ``._ast_fields`` describing
        its input args and each Funsor instance has an attribute ``._ast_args``
        with values corresponding to its input args. This allows the instance
        to be reflectively reconstructed under a different interpretation, and
        is used by :func:`funsor.interpreter.reinterpret`.
    3.  Cons-hash construction, so that repeatedly calling the constructor
        with identical args will product the same object. This enables cheap
        syntactic equality testing using the ``is`` operator, which is
        is important both for hashing (e.g. for memoizing funsor functions)
        and for unit testing, since ``.__eq__()`` is overloaded with
        elementwise semantics. Cons hashing differs from memoization in that
        it incurs no memory overhead beyond the cons hash dict.
    """
    def __init__(cls, name, bases, dct):
        super(FunsorMeta, cls).__init__(name, bases, dct)
        cls._ast_fields = getargspec(cls.__init__)[0][1:]
        cls._cons_cache = WeakValueDictionary()

    def __call__(cls, *args, **kwargs):
        # Convert kwargs to args.
        if kwargs:
            args = list(args)
            for name in cls._ast_fields[len(args):]:
                args.append(kwargs.pop(name))
            assert not kwargs, kwargs
            args = tuple(args)

        return interpret(cls, *args)


@add_metaclass(FunsorMeta)
class Funsor(object):
    """
    Abstract base class for immutable functional tensors.

    Concrete derived classes must implement ``__init__()`` methods taking
    hashable ``*args`` and no optional ``**kwargs`` so as to support cons
    hashing. Derived classes must implement an :meth:`eager_subs` method.

    :param OrderedDict inputs: A mapping from input name to domain.
        This can be viewed as a typed context or a mapping from
        free variables to domains.
    :param Domain output: An output domain.
    """
    def __init__(self, inputs, output):
        assert isinstance(inputs, OrderedDict)
        for name, input_ in inputs.items():
            assert isinstance(name, str)
            assert isinstance(input_, Domain)
        assert isinstance(output, Domain)
        super(Funsor, self).__init__()
        self.inputs = inputs
        self.output = output

    @property
    def dtype(self):
        return self.output.dtype

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, ', '.join(map(repr, self._ast_values)))

    def _pretty(self, lines, indent=0):
        lines.append((indent, type(self).__name__))
        for arg in self._ast_values:
            if isinstance(arg, Funsor):
                arg._pretty(lines, indent + 1)
            else:
                lines.append((indent + 1, str(arg)))

    def pretty(self):
        lines = []
        self._pretty(lines)
        return '\n'.join('|   ' * indent + text for indent, text in lines)

    def __call__(self, *args, **kwargs):
        """
        Partially evaluates this funsor by substituting dimensions.
        """
        # Eagerly restrict to this funsor's inputs and convert to_funsor().
        subs = OrderedDict(zip(self.inputs, args))
        for k in self.inputs:
            if k in kwargs:
                subs[k] = kwargs[k]
        for k, v in subs.items():
            if isinstance(v, str):
                # Allow renaming of inputs via syntax x(y="z").
                v = Variable(v, self.inputs[k])
            elif isinstance(v, numbers.Number):
                v = Number(v, self.inputs[k].dtype)
            else:
                v = to_funsor(v)
            if v.output != self.inputs[k]:
                raise TypeError('Expected substitution of {} to have type {}, but got {}'
                                .format(repr(k), v.output, self.inputs[k]))
            subs[k] = v
        return self.eager_subs(tuple(subs.items()))

    def __bool__(self):
        if self.inputs or self.output.shape:
            raise ValueError(
                "bool value of Funsor with more than one value is ambiguous")
        raise NotImplementedError

    def __nonzero__(self):
        return self.__bool__()

    def item(self):
        if self.inputs or self.output.shape:
            raise ValueError(
                "only one element Funsors can be converted to Python scalars")
        raise NotImplementedError

    def reduce(self, op, reduced_vars=None):
        """
        Reduce along all or a subset of inputs.

        :param callable op: A reduction operation.
        :param reduced_vars: An optional input name or set of names to reduce.
            If unspecified, all inputs will be reduced.
        :type reduced_vars: str or frozenset
        """
        # Eagerly convert reduced_vars to appropriate things.
        if reduced_vars is None:
            # Empty reduced_vars means "reduce over everything".
            reduced_vars = frozenset(self.inputs)
        elif isinstance(reduced_vars, str):
            # A single name means "reduce over this one variable".
            reduced_vars = frozenset([reduced_vars])
        assert isinstance(reduced_vars, frozenset), reduced_vars
        if not reduced_vars:
            return self
        assert reduced_vars.issubset(self.inputs)
        return Reduce(op, self, reduced_vars)

    def sample(self, sampled_vars, sample_inputs=None):
        assert isinstance(sampled_vars, frozenset)
        if sampled_vars.isdisjoint(self.inputs):
            return self
        raise NotImplementedError

    def align(self, names):
        """
        Align this funsor to match given ``names``.
        This is mainly useful in preparation for extracting ``.data``
        of a :class:`funsor.torch.Tensor`.

        :param tuple names: A tuple of strings representing all names
            but in a new order.
        :return: A permuted funsor equivalent to self.
        :rtype: Funsor
        """
        assert isinstance(names, tuple)
        if not names or names == tuple(self.inputs):
            return self
        return Align(self, names)

    @abstractmethod
    def eager_subs(self, subs):
        """
        Internal substitution function. This relies on the user-facing
        :meth:`__call__` method to coerce non-Funsors to Funsors. Once all
        inputs are Funsors, :meth:`eager_subs` implementations can recurse to
        call other :meth:`eager_subs` methods.
        """
        raise NotImplementedError

    def eager_unary(self, op):
        return None  # defer to default implementation

    def eager_reduce(self, op, reduced_vars):
        assert reduced_vars.issubset(self.inputs)  # FIXME Is this valid?
        if not reduced_vars:
            return self

        # Try to sum out integer scalars. This is mainly useful for testing,
        # since reduction is more efficiently implemented by Tensor.
        eager_vars = []
        lazy_vars = []
        for k in reduced_vars:
            if isinstance(self.inputs[k].dtype, integer_types) and not self.inputs[k].shape:
                eager_vars.append(k)
            else:
                lazy_vars.append(k)
        if eager_vars:
            result = None
            for values in itertools.product(*(self.inputs[k] for k in eager_vars)):
                subs = dict(zip(eager_vars, values))
                result = self(**subs) if result is None else op(result, self(**subs))
            if lazy_vars:
                result = Reduce(op, result, frozenset(lazy_vars))
            return result

        return None  # defer to default implementation

    # The following methods conform to a standard array/tensor interface.

    def __invert__(self):
        return Unary(ops.invert, self)

    def __neg__(self):
        return Unary(ops.neg, self)

    def abs(self):
        return Unary(ops.abs, self)

    def sqrt(self):
        return Unary(ops.sqrt, self)

    def exp(self):
        return Unary(ops.exp, self)

    def log(self):
        return Unary(ops.log, self)

    def log1p(self):
        return Unary(ops.log1p, self)

    # The following reductions are treated as Unary ops because they
    # reduce over output shape while preserving all inputs.
    # To reduce over inputs, instead call .reduce(op, reduced_vars).

    def sum(self):
        return Unary(ops.add, self)

    def prod(self):
        return Unary(ops.mul, self)

    def logsumexp(self):
        return Unary(ops.logaddexp, self)

    def all(self):
        return Unary(ops.and_, self)

    def any(self):
        return Unary(ops.or_, self)

    def min(self):
        return Unary(ops.min, self)

    def max(self):
        return Unary(ops.max, self)

    def __add__(self, other):
        return Binary(ops.add, self, to_funsor(other))

    def __radd__(self, other):
        return Binary(ops.add, self, to_funsor(other))

    def __sub__(self, other):
        return Binary(ops.sub, self, to_funsor(other))

    def __rsub__(self, other):
        return Binary(ops.sub, to_funsor(other), self)

    def __mul__(self, other):
        return Binary(ops.mul, self, to_funsor(other))

    def __rmul__(self, other):
        return Binary(ops.mul, self, to_funsor(other))

    def __truediv__(self, other):
        return Binary(ops.truediv, self, to_funsor(other))

    def __rtruediv__(self, other):
        return Binary(ops.truediv, to_funsor(other), self)

    def __pow__(self, other):
        return Binary(ops.pow, self, to_funsor(other))

    def __rpow__(self, other):
        return Binary(ops.pow, to_funsor(other), self)

    def __and__(self, other):
        return Binary(ops.and_, self, to_funsor(other))

    def __rand__(self, other):
        return Binary(ops.and_, self, to_funsor(other))

    def __or__(self, other):
        return Binary(ops.or_, self, to_funsor(other))

    def __ror__(self, other):
        return Binary(ops.or_, self, to_funsor(other))

    def __xor__(self, other):
        return Binary(ops.xor, self, to_funsor(other))

    def __eq__(self, other):
        return Binary(ops.eq, self, to_funsor(other))

    def __ne__(self, other):
        return Binary(ops.ne, self, to_funsor(other))

    def __lt__(self, other):
        return Binary(ops.lt, self, to_funsor(other))

    def __le__(self, other):
        return Binary(ops.le, self, to_funsor(other))

    def __gt__(self, other):
        return Binary(ops.gt, self, to_funsor(other))

    def __ge__(self, other):
        return Binary(ops.ge, self, to_funsor(other))

    def __min__(self, other):
        return Binary(ops.min, self, to_funsor(other))

    def __max__(self, other):
        return Binary(ops.max, self, to_funsor(other))

    def __getitem__(self, other):
        return Binary(ops.getitem, self, to_funsor(other))


interpreter.reinterpret.register(Funsor)(interpreter.reinterpret_funsor)


@singledispatch
def to_funsor(x):
    """
    Convert to a :class:`Funsor`.
    Only :class:`Funsor`s and scalars are accepted.

    :param x: An object.
    :return: A Funsor equivalent to ``x``.
    :rtype: Funsor
    :raises: ValueError
    """
    raise ValueError("cannot convert to Funsor: {}".format(x))


@to_funsor.register(Funsor)
def _to_funsor_funsor(x):
    return x


class Variable(Funsor):
    """
    Funsor representing a single free variable.

    :param str name: A variable name.
    :param funsor.domains.Domain output: A domain.
    """
    def __init__(self, name, output):
        inputs = OrderedDict([(name, output)])
        super(Variable, self).__init__(inputs, output)
        self.name = name

    def __repr__(self):
        return "Variable({}, {})".format(repr(self.name), repr(self.output))

    def __str__(self):
        return self.name

    def eager_subs(self, subs):
        assert isinstance(subs, tuple)
        for k, v in subs:
            if k == self.name:
                return v
        return self


_PREFIX = {
    ops.neg: '-',
    ops.invert: '~',
}


class Unary(Funsor):
    """
    Lazy unary operation.
    """
    def __init__(self, op, arg):
        assert callable(op)
        assert isinstance(arg, Funsor)
        output = find_domain(op, arg.output)
        super(Unary, self).__init__(arg.inputs, output)
        self.op = op
        self.arg = arg

    def __repr__(self):
        if self.op in _PREFIX:
            return '{}{}'.format(_PREFIX[self.op], self.arg)
        return 'Unary({}, {})'.format(self.op.__name__, self.arg)

    def eager_subs(self, subs):
        if not any(k in self.inputs for k, v in subs):
            return self
        arg = self.arg.eager_subs(subs)
        return Unary(self.op, arg)


@eager.register(Unary, Op, Funsor)
def eager_unary(op, arg):
    return arg.eager_unary(op)


_INFIX = {
    ops.add: '+',
    ops.sub: '-',
    ops.mul: '*',
    ops.truediv: '/',
    ops.pow: '**',
}


class Binary(Funsor):
    """
    Lazy binary operation.
    """
    def __init__(self, op, lhs, rhs):
        assert callable(op)
        assert isinstance(lhs, Funsor)
        assert isinstance(rhs, Funsor)
        inputs = lhs.inputs.copy()
        inputs.update(rhs.inputs)
        output = find_domain(op, lhs.output, rhs.output)
        super(Binary, self).__init__(inputs, output)
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        if self.op in _INFIX:
            return '({} {} {})'.format(self.lhs, _INFIX[self.op], self.rhs)
        return 'Binary({}, {}, {})'.format(self.op.__name__, self.lhs, self.rhs)

    def eager_subs(self, subs):
        if not any(k in self.inputs for k, v in subs):
            return self
        lhs = self.lhs.eager_subs(subs)
        rhs = self.rhs.eager_subs(subs)
        return Binary(self.op, lhs, rhs)


class Reduce(Funsor):
    """
    Lazy reduction over multiple variables.
    """
    def __init__(self, op, arg, reduced_vars):
        assert callable(op)
        assert isinstance(arg, Funsor)
        assert isinstance(reduced_vars, frozenset)
        inputs = OrderedDict((k, v) for k, v in arg.inputs.items() if k not in reduced_vars)
        output = arg.output
        super(Reduce, self).__init__(inputs, output)
        self.op = op
        self.arg = arg
        self.reduced_vars = reduced_vars

    def __repr__(self):
        return 'Reduce({}, {}, {})'.format(
            self.op.__name__, self.arg, self.reduced_vars)

    def eager_subs(self, subs):
        subs = tuple((k, v) for k, v in subs if k not in self.reduced_vars)
        if not any(k in self.inputs for k, v in subs):
            return self
        if not all(self.reduced_vars.isdisjoint(v.inputs) for k, v in subs):
            raise NotImplementedError('TODO alpha-convert to avoid conflict')
        return self.arg.eager_subs(subs).reduce(self.op, self.reduced_vars)

    def eager_reduce(self, op, reduced_vars):
        if op is self.op:
            # Eagerly fuse reductions.
            assert isinstance(reduced_vars, frozenset)
            reduced_vars = reduced_vars.intersection(self.inputs) | self.reduced_vars
            return Reduce(op, self.arg, reduced_vars)
        return super(Reduce, self).reduce(op, reduced_vars)


@eager.register(Reduce, AssociativeOp, Funsor, frozenset)
def eager_reduce(op, arg, reduced_vars):
    return arg.eager_reduce(op, reduced_vars)


class NumberMeta(FunsorMeta):
    """
    Wrapper to fill in default ``dtype``.
    """
    def __call__(cls, data, dtype=None):
        if dtype is None:
            dtype = "real"
        return super(NumberMeta, cls).__call__(data, dtype)


@to_funsor.register(numbers.Number)
@add_metaclass(NumberMeta)
class Number(Funsor):
    """
    Funsor backed by a Python number.

    :param numbers.Number data: A python number.
    :param dtype: A nonnegative integer or the string "real".
    """
    def __init__(self, data, dtype=None):
        assert isinstance(data, numbers.Number)
        if isinstance(dtype, integer_types):
            data = type(dtype)(data)
        else:
            assert isinstance(dtype, str) and dtype == "real"
            data = float(data)
        inputs = OrderedDict()
        output = Domain((), dtype)
        super(Number, self).__init__(inputs, output)
        self.data = data

    def __repr__(self):
        if self.dtype == "real":
            return 'Number({}, "real")'.format(repr(self.data))
        else:
            return 'Number({}, {})'.format(repr(self.data), self.dtype)

    def __str__(self):
        return str(self.data)

    def __int__(self):
        return int(self.data)

    def __float__(self):
        return float(self.data)

    def __bool__(self):
        return bool(self.data)

    def item(self):
        return self.data

    def eager_subs(self, subs):
        return self

    def eager_unary(self, op):
        return Number(op(self.data), self.dtype)


@eager.register(Binary, Op, Number, Number)
def eager_binary_number_number(op, lhs, rhs):
    data = op(lhs.data, rhs.data)
    output = find_domain(op, lhs.output, rhs.output)
    dtype = output.dtype
    return Number(data, dtype)


class Align(Funsor):
    """
    Lazy call to ``.align(...)``.
    """
    def __init__(self, arg, names):
        assert isinstance(arg, Funsor)
        assert isinstance(names, tuple)
        assert all(isinstance(name, str) for name in names)
        assert all(name in arg.inputs for name in names)
        inputs = OrderedDict((name, arg.inputs[name]) for name in names)
        inputs.update(arg.inputs)
        output = arg.output
        super(Align, self).__init__(inputs, output)
        self.arg = arg

    def align(self, names):
        return self.arg.align(names)

    def eager_subs(self, subs):
        assert isinstance(subs, tuple)
        return self.arg.eager_subs(subs)

    def eager_unary(self, op):
        return self.arg.eager_unary(op)

    def eager_reduce(self, op, reduced_vars):
        return self.arg.eager_reduce(op, reduced_vars)


@eager.register(Binary, Op, Align, Funsor)
def eager_binary_align_funsor(op, lhs, rhs):
    return Binary(op, lhs.arg, rhs)


@eager.register(Binary, Op, Funsor, Align)
def eager_binary_funsor_align(op, lhs, rhs):
    return Binary(op, lhs, rhs.arg)


@eager.register(Binary, Op, Align, Align)
def eager_binary_align_align(op, lhs, rhs):
    return Binary(op, lhs.arg, rhs.arg)


class Stack(Funsor):
    """
    Stack of funsors along a new input dimension.

    :param tuple components: A tuple of Funsors.
    :param str name: The name of the new leftmost dimension.
    """
    def __init__(self, components, name):
        assert isinstance(components, tuple)
        assert components
        assert not any(name in x.inputs for x in components)
        assert len(set(x.output for x in components)) == 1
        output = components[0].output
        domain = bint(len(components))
        inputs = OrderedDict([(name, domain)])
        for x in components:
            inputs.update(x.inputs)
        super(Stack, self).__init__(inputs, output)
        self.components = components
        self.name = name

    def eager_subs(self, subs):
        assert isinstance(subs, tuple)
        if not any(k in self.inputs for k, v in subs):
            return self
        pos = None
        for i, (k, index) in enumerate(subs):
            if k == self.name:
                pos = i
                break

        if pos is None:
            # Eagerly recurse into components.
            assert not any(self.name in v.inputs for k, v in subs)
            components = tuple(x.eager_subs(subs) for x in self.components)
            return Stack(components, self.name)

        # Try to eagerly select an index.
        assert index.output == bint(len(self.components))
        subs = subs[:pos] + subs[1 + pos:]

        if isinstance(index, Number):
            # Select a single component.
            result = self.components[index.data]
            return result.eager_subs(subs)

        if isinstance(index, Variable):
            # Rename the stacking dimension.
            components = self.components
            if subs:
                components = tuple(x.eager_subs(subs) for x in components)
            return Stack(components, index.name)

        if not subs:
            raise NotImplementedError('TODO support advanced indexing in Stack')

        # Eagerly recurse into components but lazily substitute.
        components = tuple(x.eager_subs(subs) for x in self.components)
        result = Stack(components, self.name)
        return result.eager_subs(((self.name, index),))

    def eager_reduce(self, op, reduced_vars):
        components = self.components
        if self.name in reduced_vars:
            reduced_vars -= frozenset([self.name])
            if reduced_vars:
                components = tuple(x.reduce(op, reduced_vars) for x in components)
            return reduce(op, components)
        components = tuple(x.reduce(op, reduced_vars) for x in components)
        return Stack(components, self.name)


def _of_shape(fn, shape):
    args, vargs, kwargs, defaults = getargspec(fn)
    assert not vargs
    assert not kwargs
    names = tuple(args)
    args = [Variable(name, size) for name, size in zip(names, shape)]
    return to_funsor(fn(*args)).align(names)


def of_shape(*shape):
    """
    Decorator to construct a :class:`Funsor` with one free :class:`Variable`
    per function arg.
    """
    return functools.partial(_of_shape, shape=shape)


__all__ = [
    'Binary',
    'Funsor',
    'Number',
    'Reduce',
    'Stack',
    'Unary',
    'Variable',
    'eager',
    'lazy',
    'of_shape',
    'reflect',
    'to_funsor',
]
