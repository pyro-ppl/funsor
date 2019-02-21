from __future__ import absolute_import, division, print_function

import itertools
import numbers
from collections import OrderedDict

from six import add_metaclass
from six.moves import ABCMeta, abstractmethod

import funsor.ops as ops
from funsor.domains import Domain, find_domain
from funsor.interpretations import eager, get_interpretation, lazy, reflect
from funsor.six import getargspec, singledispatch


class FunsorMeta(ABCMeta):
    def __init__(cls, name, bases, dct):
        super(FunsorMeta, cls).__init__(name, bases, dct)
        cls._ast_fields = getargspec(cls.__init__)[0][1:]

    def __call__(cls, *args, **kwargs):
        # Convert kwargs to args.
        if kwargs:
            args = list(args)
            for name in cls._ast_fields[len(args):]:
                args.append(kwargs.pop(name))
            assert not kwargs, kwargs
            args = tuple(args)

        # Apply interpretation.
        result = get_interpretation()(cls, *args)
        if result is not None:
            return result

        # Create a new object.
        result = super(FunsorMeta, cls).__call__(*args)
        result._ast_values = args
        return result


@add_metaclass(FunsorMeta)
class Funsor(object):
    """
    Abstract base class for immutable functional tensors.

    Derived classes must implement an :meth:`eager_subs` method.

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
        super(Funsor, self).__init__(inputs, output)
        self.inputs = inputs
        self.output = output

    def __hash__(self):
        return id(self)

    def __call__(self, *args, **kwargs):
        """
        Partially evaluates this funsor by substituting dimensions.
        """
        # Eagerly restrict to this funsor's inputs and convert to_funsor().
        subs = OrderedDict(zip(self.inputs, args))
        for k in self.inputs[len(args):]:
            if k in kwargs:
                subs[k] = kwargs[k]
        for k, v in subs.items():
            if isinstance(v, str):
                subs[v] = Variable(v, self.inputs[v])
            else:
                subs[v] = to_funsor(v)
        return Substitute(self, subs)

    def __getitem__(self, args):
        if not isinstance(args, tuple):
            args = (args,)

        # Handle Ellipsis notation like x[..., 0].
        kwargs = {}
        for pos, arg in enumerate(args):
            if arg is Ellipsis:
                kwargs.update(zip(reversed(self.inputs),
                                  reversed(args[1 + pos:])))
                break
            kwargs[self.inputs[pos]] = arg  # FIXME fix positional indexing

        # Handle complete slices like x[:].
        kwargs = {key: value
                  for key, value in kwargs.items()
                  if not (isinstance(value, slice) and value == slice(None))}

        return self(**kwargs)

    # Avoid __setitem__ due to immutability.

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

    def reduce(self, op, bound_vars=None):
        """
        Reduce along all or a subset of inputs.

        :param callable op: A reduction operation.
        :param bound_vars: An optional input name or set of names to reduce.
            If unspecified, all inputs will be reduced.
        :type bound_vars: str or frozenset
        """
        # Eagerly convert bound_vars to appropriate things.
        if bound_vars is None:
            # Empty bound_vars means "reduce over everything".
            bound_vars = frozenset(self.inputs)
        else:
            # A single name means "reduce over this one variable".
            if isinstance(bound_vars, str):
                bound_vars = frozenset([bound_vars])
            assert bound_vars.issubset(self.inputs)
        return Reduce(op, self, bound_vars)

    @abstractmethod
    def eager_subs(self, subs):
        raise NotImplementedError

    def eager_unary(self, op):
        return None  # defer to default implementation

    def eager_reduce(self, op, bound_vars):
        assert all(k in self.inputs for k in bound_vars)
        if not bound_vars:
            return self

        # Try to sum out integer variables. This is mainly useful for testing,
        # since reduction is more efficiently implemented by Tensor.
        int_vars = tuple(sorted(k for k in bound_vars
                                if isinstance(self.inputs[k].dtype, int)))
        if int_vars:
            result = 0.
            for int_values in itertools.product(*(self.inputs[k] for k in int_vars)):
                subs = dict(zip(int_vars, int_values))
                result += self(**subs)
            return result

        return None  # defer to default implementation

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

    def sum(self, bound_vars=None):
        return Reduce(ops.add, self, bound_vars)

    def prod(self, bound_vars=None):
        return Reduce(ops.mul, self, bound_vars)

    def logsumexp(self, bound_vars=None):
        return Reduce(ops.logaddexp, self, bound_vars)

    def all(self, bound_vars=None):
        return Reduce(ops.and_, self, bound_vars)

    def any(self, bound_vars=None):
        return Reduce(ops.or_, self, bound_vars)

    def min(self, bound_vars=None):
        return Reduce(ops.min, self, bound_vars)

    def max(self, bound_vars=None):
        return Reduce(ops.max, self, bound_vars)


@singledispatch
def to_funsor(x):
    """
    Convert to a :class:`Funsor`.
    Only :class:`Funsor`s and scalars are accepted.
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
        return "Variable({}, {})".format(repr(self.name), repr(self.shape[0]))

    def __str__(self):
        return self.name

    def eager_subs(self, subs):
        return subs.get(self.name, self)


class Substitute(Funsor):
    """
    Lazy substitution of the form ``x(u=y, v=z)``.
    """
    def __init__(self, arg, subs):
        assert isinstance(arg, Funsor)
        assert isinstance(subs, dict)
        for key, value in subs.items():
            assert isinstance(key, str)
            assert key in arg.inputs
            assert isinstance(value, Funsor)
        inputs = arg.inputs.copy()
        for key, value in subs:
            del inputs[key]
        for key, value in subs:
            inputs.update(value.inputs)
        super(Substitute, self).__init__(inputs, arg.output)
        self.arg = arg
        self.subs = subs

    def __repr__(self):
        return 'Substitute({}, {})'.format(self.arg, self.subs)

    def eager_subs(self, subs):
        raise NotImplementedError('TODO')


@lazy.register(Substitute, Funsor, object)
@eager.register(Substitute, Funsor, object)
def eager_subs(arg, subs):
    if set(subs).isdisjoint(arg.inputs):
        return arg
    return arg.eager_subs(subs)


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
        arg = eager_subs(self.arg, subs)
        return Unary(self.op, arg)


lazy.register(Unary, object, Funsor)(reflect)


@eager.register(Unary, object, Funsor)
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
        lhs = eager_subs(self.lhs, subs)
        rhs = eager_subs(self.rhs, subs)
        return Binary(self.op, lhs, rhs)


lazy.register(Binary, object, Funsor, Funsor)(lazy)
eager.register(Binary, object, Funsor, Funsor)(lazy)


class Reduce(Funsor):
    """
    Lazy reduction.
    """
    def __init__(self, op, arg, bound_vars):
        assert callable(op)
        assert isinstance(arg, Funsor)
        assert isinstance(bound_vars, frozenset)
        inputs = OrderedDict((k, v) for k, v in arg.inputs.items() if k not in bound_vars)
        output = arg.output
        super(Reduce, self).__init__(inputs, output)
        self.op = op
        self.arg = arg
        self.bound_vars = bound_vars

    def __repr__(self):
        return 'Reduce({}, {}, {})'.format(
            self.op.__name__, self.arg, self.bound_vars)

    def eager_subs(self, subs):
        subs = {key: value for key, value in subs.items() if key not in self.bound_vars}
        if not all(self.bound_vars.isdisjoint(value.inputs)
                   for value in subs.values()):
            raise NotImplementedError('TODO alpha-convert to avoid conflict')
        return self.arg(**subs).reduce(self.op, self.bound_vars)

    def eager_reduce(self, op, bound_vars=None):
        if op is self.op:
            # Eagerly fuse reductions.
            if bound_vars is None:
                bound_vars = frozenset(self.bound_vars)
            else:
                bound_vars = frozenset(bound_vars).intersection(self.bound_vars)
            return Reduce(op, self.arg, self.bound_vars | bound_vars)
        return super(Reduce, self).reduce(op, bound_vars)


lazy.register(Reduce, object, Funsor, frozenset)(reflect)


@eager.register(Reduce, object, Funsor, frozenset)
def eager_reduce(op, arg, bound_vars):
    return arg.eager_reduce(op, bound_vars)


@to_funsor.register(float)
class Number(Funsor):
    """
    Funsor backed by a Python number.

    :param numbers.Number data: A python number.
    :param dtype: A nonnegative integer or the string "real".
    """
    def __init__(self, data, dtype="real"):
        assert isinstance(data, numbers.Number)
        if isinstance(dtype, int):
            data = int(data)
        else:
            assert isinstance(dtype, str) and dtype == "real"
            data = float(data)
        inputs = OrderedDict()
        output = Domain((), dtype)
        super(Number, self).__init__(inputs, output)
        self.data = data

    def __repr__(self):
        if self.output.dtype == "real":
            return 'Number({}, "real")'.format(repr(self.data))
        else:
            return 'Number({}, {})'.format(repr(self.data), self.output.dtype)

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
        return Number(op(self.data))


@eager.register(Binary, object, Number, Number)
def eager_binary_number_number(op, lhs, rhs):
    return Number(op(lhs.data, rhs.data))


__all__ = [
    'Binary',
    'Funsor',
    'Number',
    'Reduce',
    'Substitute',
    'Unary',
    'Variable',
    'to_funsor',
]
