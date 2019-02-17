from __future__ import absolute_import, division, print_function

import functools
import numbers
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

from multipledispatch import dispatch
from six import add_metaclass

import funsor.ops as ops
from funsor.handlers import effectful
from funsor.interpretations import Eager, Simplify
from funsor.six import getargspec, singledispatch

DOMAINS = ('real', 'vector')


class FunsorMeta(ABCMeta):

    def __init__(cls, name, bases, dct):
        super(FunsorMeta, cls).__init__(name, bases, dct)
        cls._ast_fields = getargspec(cls.__init__)[0][1:]

    def __call__(cls, *args, **kwargs):
        # TODO move this back into Memoize?
        # Convert kwargs to args.
        if kwargs:
            args = list(args)
            for name in cls._ast_fields[len(args):]:
                args.append(kwargs.pop(name))
            assert not kwargs, kwargs
            args = tuple(args)

        return effectful(cls, cls._ast_call)(*args)

    def _ast_call(cls, *args):
        result = super(FunsorMeta, cls).__call__(*args)
        result._ast_values = args
        return result


@add_metaclass(FunsorMeta)
class Funsor(object):
    """
    Abstract base class for immutable functional tensors.

    Concrete derived classes must implement ``__init__()`` methods taking
    hashable ``*args`` and no optional ``**kwargs`` so as to support cons
    hashing. Concrete derive classes must also implement an ``_eager_subs()``
    method defining substitution.

    .. note:: Probabilistic methods like :meth:`sample` and :meth:`marginal`
        follow the convention that funsors represent log density functions.
        Thus for example the partition function is given by :meth:`logsumexp`.

    :ivar OrderedDict schema: A mapping from dim to size.
    :param tuple dims: A tuple of strings of dimension names.
    :param tuple shape: A tuple of sizes. Each size is either a nonnegative
        integer or a string denoting a continuous domain.
    """
    def __init__(self, dims, shape):
        assert isinstance(dims, tuple)
        assert all(isinstance(d, str) for d in dims)
        assert len(set(dims)) == len(dims)
        assert isinstance(shape, tuple)
        assert all(isinstance(s, int) or s in DOMAINS for s in shape)
        assert len(dims) == len(shape)
        super(Funsor, self).__init__()
        self.dims = dims
        self.shape = shape
        self.schema = OrderedDict(zip(dims, shape))

    def __hash__(self):
        return id(self)

    def __call__(self, *args, **kwargs):
        """
        Partially evaluates this funsor by substituting dimensions.
        """
        subs = OrderedDict(zip(self.dims, args))
        for d in self.dims:
            if d in kwargs:
                subs[d] = kwargs[d]
        return Substitution(self, tuple((k, to_funsor(v)) for k, v in subs.items()))

    @abstractmethod
    def _eager_subs(self, **kwargs):
        raise NotImplementedError

    def __getitem__(self, args):
        if not isinstance(args, tuple):
            args = (args,)

        # Handle Ellipsis notation like x[..., 0].
        kwargs = {}
        for pos, arg in enumerate(args):
            if arg is Ellipsis:
                kwargs.update(zip(reversed(self.dims),
                                  reversed(args[1 + pos:])))
                break
            kwargs[self.dims[pos]] = arg

        # Handle complete slices like x[:].
        kwargs = {dim: value
                  for dim, value in kwargs.items()
                  if not (isinstance(value, slice) and value == slice(None))}

        return self(**kwargs)

    # Avoid __setitem__ due to immutability.

    def align(self, dims, shape=None):
        """
        Align this funsor to match given ``dims`` and ``shape``.

        This can both permute and add constant dims.  This is mainly useful in
        preparation for extracting ``.data`` of a :class:`funsor.torch.Tensor`.
        """
        if shape is None:
            assert set(dims) == set(self.dims)
            shape = tuple(self.schema[d] for d in dims)
        if dims == self.dims:
            assert shape == self.shape
            return self
        return Align(self, dims, shape)

    def __bool__(self):
        if self.shape:
            raise ValueError(
                "bool value of Funsor with more than one value is ambiguous")
        raise NotImplementedError

    def __nonzero__(self):
        return self.__bool__()

    def item(self):
        if self.shape:
            raise ValueError(
                "only one element Funsors can be converted to Python scalars")
        raise NotImplementedError

    def _eager_unary(self, op):
        pass  # defer to lazy operation

    def reduce(self, op, dims=None):
        """
        Reduce along all or a subset of dimensions.

        :param callable op: A reduction operation.
        :param set dims: An optional dim or set of dims to reduce.
            If unspecified, all dims will be reduced.
        """
        if dims is None:
            dims = frozenset(self.dims)
        else:
            if isinstance(dims, str):
                dims = (dims,)
            dims = frozenset(dims).intersection(self.dims)
        if not dims:
            return self
        return Reduction(op, self, dims)

    def _eager_reduce(self, op, dims):
        pass  # defer to lazy operation

    def contract(self, sum_op, prod_op, other, dims):
        """
        DEPRECATED This is equivalent to

            (self * other).reduce(sum_op, dims)

        but we only want to perform eager contractions.
        """
        raise NotImplementedError(
            "contract({}, {}, {}, {}, {})".format(sum_op.__name__, prod_op.__name__, self, other, dims))

    def jacobian(self, dim):
        if dim not in self.dims:
            return Number(0.)
        raise NotImplementedError

    # ------------------------------------------------------------------------
    # Subclasses should not override these methods; instead override
    # the generic handlers and fall back to super(...).handler.

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

    def sum(self, dims=None):
        return self.reduce(ops.add, dims)

    def prod(self, dims=None):
        return self.reduce(ops.mul, dims)

    def logsumexp(self, dims=None):
        return self.reduce(ops.logaddexp, dims)

    def all(self, dims=None):
        return self.reduce(ops.and_, dims)

    def any(self, dims=None):
        return self.reduce(ops.or_, dims)

    def min(self, dims=None):
        return self.reduce(ops.min, dims)

    def max(self, dims=None):
        return self.reduce(ops.max, dims)


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
    :param size: A size, either an int or a ``DOMAIN``.
    """
    def __init__(self, name, size):
        super(Variable, self).__init__((name,), (size,))

    def __repr__(self):
        return "Variable({}, {})".format(repr(self.name), repr(self.shape[0]))

    def __str__(self):
        return self.name

    @property
    def name(self):
        return self.dims[0]

    def _eager_subs(self, **kwargs):
        return kwargs.get(self.name, self)

    def jacobian(self, dim):
        return Number(float(dim == self.name))


class Substitution(Funsor):
    """
    Lazy substitution of the form ``x(u=y, v=z)``.
    """
    def __init__(self, arg, subs):
        assert isinstance(arg, Funsor)
        assert isinstance(subs, tuple)
        for dim_value in subs:
            assert isinstance(dim_value, tuple)
            dim, value = dim_value
            assert isinstance(dim, str)
            assert dim in arg.dims
            assert isinstance(value, Funsor)
        schema = arg.schema.copy()
        for dim, value in subs:
            del schema[dim]
        for dim, value in subs:
            schema.update(value.schema)
        dims = tuple(schema)
        shape = tuple(schema.values())
        super(Substitution, self).__init__(dims, shape)
        self.arg = arg
        self.subs = subs

    def __repr__(self):
        return 'Substitution({}, {})'.format(self.arg, self.subs)

    def _eager_subs(self, **kwargs):
        # TODO eagerly fuse substitutions.
        subs = {dim: value(**kwargs) for dim, value in self.subs}
        for dim, value in self.subs:
            kwargs.pop(dim, None)
        result = self.arg(**kwargs)(**subs)
        # FIXME for densities, add log_abs_det_jacobian
        return result


@Eager.register(Substitution)
def _eager_subs(arg, subs):
    return arg._eager_subs(**dict(subs))


class Align(Funsor):
    """
    Lazy call to ``.align(...)``.

    Operations on :class:`Align` funsors do not preserve alignment.
    """
    def __init__(self, arg, dims, shape):
        assert isinstance(arg, Funsor)
        assert isinstance(dims, tuple)
        assert isinstance(shape, tuple)
        assert all(isinstance(d, str) for d in dims)
        assert all(isinstance(s, int) or s in DOMAINS for s in shape)
        for d, s in zip(dims, shape):
            assert arg.schema.get(d, s) == s
        super(Align, self).__init__(dims, shape)
        self.arg = arg
        self.dims = dims
        self.shape = shape

    def _eager_subs(self, **kwargs):
        return self.arg(**kwargs)

    def align(self, dims, shape=None):
        if shape is None:
            assert set(dims) == set(self.dims)
            shape = tuple(self.schema[d] for d in dims)
        return self.arg.align(dims, shape)

    def _eager_unary(self, op):
        return self.arg._eager_unary(op)

    def _eager_reduce(self, op, dims):
        return self.arg.reduce(op, dims)


@dispatch(object, Align, Funsor)
def eager_binary(op, lhs, rhs):
    return Binary(op, lhs.arg, rhs)


@dispatch(object, Funsor, Align)
def eager_binary(op, lhs, rhs):
    return Binary(op, lhs, rhs.arg)


@dispatch(object, Align, Align)
def eager_binary(op, lhs, rhs):
    return Binary(op, lhs.arg, rhs.arg)


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
        super(Unary, self).__init__(arg.dims, arg.shape)
        self.op = op
        self.arg = arg

    def __repr__(self):
        if self.op in _PREFIX:
            return '{}{}'.format(_PREFIX[self.op], self.arg)
        return 'Unary({}, {})'.format(self.op.__name__, self.arg)

    def _eager_subs(self, **kwargs):
        return Unary(self.op, self.arg(**kwargs))

    def _simplify_unary(self, op):
        if op is ops.neg and self.op is ops.neg:
            return self.arg

    def jacobian(self, dim):
        if dim not in self.arg.dims:
            return Number(0.)
        if self.op is ops.neg:
            return -self.arg.jacobian(dim)
        raise NotImplementedError


@Eager.register(Unary)
def _eager_unary(op, arg):
    return arg._eager_unary(op)


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
        schema = lhs.schema.copy()
        schema.update(rhs.schema)
        dims = tuple(schema)
        shape = tuple(schema.values())
        super(Binary, self).__init__(dims, shape)
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        if self.op in _INFIX:
            return '({} {} {})'.format(self.lhs, _INFIX[self.op], self.rhs)
        return 'Binary({}, {}, {})'.format(self.op.__name__, self.lhs, self.rhs)

    def _eager_subs(self, **kwargs):
        return Binary(self.op, self.lhs(**kwargs), self.rhs(**kwargs))

    def contract(self, sum_op, prod_op, other, dims):
        if dims not in self.dims:
            return self * other
        if prod_op is self.op:
            if dims not in self.lhs:
                return Binary(prod_op, self.lhs, self.rhs.reduce(sum_op, dims))
            if dims not in self.rhs:
                return Binary(prod_op, self.lhs.reduce(sum_op, dims), self.rhs)
        return super(Binary, self).contract(sum_op, prod_op, other, dims)

    def jacobian(self, dim):
        if dim not in self.dims:
            return Number(0.)
        if self.op is ops.add:
            return self.lhs.jacobian(dim) + self.rhs.jacobian(dim)
        if self.op is ops.sub:
            return self.lhs.jacobian(dim) - self.rhs.jacobian(dim)
        if self.op is ops.mul:
            return self.lhs.jacobian(dim) * self.rhs + self.lhs * self.rhs.jacobian(dim)
        if self.op is ops.truediv:
            if dim not in self.rhs:
                return self.lhs.jacobian(dim) / self.rhs
            return (self.lhs.jacobian(dim) * self.rhs - self.lhs * self.rhs.jacobian(dim)) / self.rhs ** 2
        raise NotImplementedError


@dispatch(object, Funsor, Funsor)
def eager_binary(op, lhs, rhs):
    pass  # defer to lazy operation


@Eager.register(Binary)
def _eager_binary(op, lhs, rhs):
    return eager_binary(op, lhs, rhs)


@dispatch(object, Funsor, Funsor)
def simplify_binary(op, lhs, rhs):
    pass  # defer to lazy operation


@Simplify.register(Binary)
def _simplify_binary(op, lhs, rhs):
    return simplify_binary(op, lhs, rhs)


class Reduction(Funsor):
    """
    Lazy reduction.
    """
    def __init__(self, op, arg, reduce_dims):
        assert callable(op)
        assert isinstance(arg, Funsor)
        assert isinstance(reduce_dims, frozenset)
        dims = tuple(d for d in arg.dims if d not in reduce_dims)
        shape = tuple(arg.schema[d] for d in dims)
        super(Reduction, self).__init__(dims, shape)
        self.op = op
        self.arg = arg
        self.reduce_dims = reduce_dims

    def __repr__(self):
        return 'Reduction({}, {}, {})'.format(
            self.op.__name__, self.arg, self.reduce_dims)

    def _eager_subs(self, **kwargs):
        kwargs = {dim: value for dim, value in kwargs.items()
                  if dim in self.dims}
        if not all(set(self.reduce_dims).isdisjoint(getattr(value, 'dims', ()))
                   for value in kwargs.values()):
            raise NotImplementedError('TODO alpha-convert to avoid conflict')
        return Reduction(self.op, self.arg(**kwargs), self.reduce_dims)

    def _eager_reduce(self, op, dims):
        if op is self.op:
            # Eagerly fuse reductions.
            dims = frozenset(dims).intersection(self.dims)
            return self.arg.reduce(op, self.reduce_dims | dims)


@Eager.register(Reduction)
def _eager_reduce(arg, op, dims):
    return arg._eager_reduce(op, dims)


class Branch(Funsor):
    """
    Funsor representing a multi-way branch statement.

    This serves as a ragged equivalent to :func:`torch.stack`.
    This is useful for modeling heterogeneous mixture models.

    Note that while dims may differ across branches, types must agree across
    branches.

    :param str dim: A dim on which to branch.
    :param tuple components: An tuple of components of heterogeneous shape.
    """
    def __init__(self, dim, components):
        assert isinstance(dim, str)
        assert isinstance(components, tuple)
        assert all(isinstance(c, Funsor) for c in components)
        schema = OrderedDict([(dim, len(components))])
        for i, c in enumerate(components):
            if dim in c.dims:
                c = c(**{dim: i})
            schema.update(c.schema)
        dims = tuple(schema)
        shape = tuple(schema.values())
        super(Branch, self).__init__(dims, shape)
        self.components = components

    @property
    def dim(self):
        return self.dims[0]

    def _eager_subs(self, **kwargs):
        # Try eagerly slicing.
        choice = kwargs.pop(self.dim, None)
        try:
            choice = int(choice)
        except (TypeError, ValueError):
            pass
        if isinstance(choice, int):
            return self.components[choice](**kwargs)

        # Try eagerly renaming self.dim.
        result = self
        if isinstance(choice, Variable):
            assert choice.shape[0] == len(self.components)
            result = Branch(choice.name, self.components)
            choice = None

        # Eagerly substitute into components, but lazily slice.
        if kwargs:
            components = tuple(c(**kwargs) for c in self.components)
            result = Branch(self.dim, components)
        if choice is not None:
            result = Substitution(result, ((self.dim, choice),))
        return result


class Finitary(Funsor):
    """
    Commutative binary operator applied to arbitrary number of operands.
    Used in the engine to rewrite term graphs to optimized forms.
    """
    def __init__(self, op, operands):
        assert callable(op)
        assert isinstance(operands, tuple)
        assert all(isinstance(operand, Funsor) for operand in operands)
        schema = OrderedDict()
        for operand in operands:
            schema.update(operand.schema)
        dims = tuple(schema)
        shape = tuple(schema.values())
        super(Finitary, self).__init__(dims, shape)
        self.op = op
        self.operands = operands

    def __repr__(self):
        return 'Finitary({}, {})'.format(self.op.__name__, self.operands)

    def _eager_subs(self, **kwargs):
        return Finitary(self.op, tuple(x(**kwargs) for x in self.operands))


class AddTypeMeta(FunsorMeta):
    def __call__(cls, data, dtype=None):
        if dtype is None:
            dtype = type(data)
        return super(AddTypeMeta, cls).__call__(data, dtype)


@to_funsor.register(numbers.Number)
@add_metaclass(AddTypeMeta)
class Number(Funsor):
    """
    Funsor backed by a Python number.

    :param numbers.Number data: A python number.
    """
    def __init__(self, data, dtype=None):
        assert isinstance(data, numbers.Number)
        assert dtype == type(data)
        super(Number, self).__init__((), ())
        self.data = data

    def __repr__(self):
        return 'Number({})'.format(repr(self.data))

    def __str__(self):
        return str(self.data)

    def __int__(self):
        return int(self.data)

    def __float__(self):
        return float(self.data)

    def _eager_subs(self, **kwargs):
        return self

    def __bool__(self):
        return bool(self.data)

    def item(self):
        return self.data

    def _eager_unary(self, op):
        return Number(op(self.data))


@dispatch(object, Number, Number)
def eager_binary(op, lhs, rhs):
    return Number(op(lhs.data, rhs.data))


@dispatch(object, Number, Funsor)
def simplify_binary(op, lhs, rhs):
    if op is ops.add:
        if lhs.data == 0:
            return rhs
    elif op is ops.sub:
        if lhs.data == 0:
            return -rhs
    elif op is ops.mul:
        if lhs.data == 1:
            return rhs


@dispatch(object, Funsor, Number)
def simplify_binary(op, lhs, rhs):
    if op is ops.add or op is ops.sub:
        if rhs.data == 0:
            return lhs
    elif op is ops.mul or op is ops.truediv:
        if rhs.data == 1:
            return lhs


@dispatch(object, Number, Number)
def simplify_binary(op, lhs, rhs):
    return Number(op(lhs.data, rhs.data))


def _of_shape(fn, shape):
    args, vargs, kwargs, defaults = getargspec(fn)
    assert not vargs
    assert not kwargs
    dims = tuple(args)
    args = [Variable(dim, size) for dim, size in zip(dims, shape)]
    return to_funsor(fn(*args)).align(dims, shape)


def of_shape(*shape):
    """
    Decorator to construct a :class:`Funsor` with one free :class:`Variable`
    per function arg.
    """
    return functools.partial(_of_shape, shape=shape)


__all__ = [
    'Align',
    'Binary',
    'Branch',
    'DOMAINS',
    'Funsor',
    'Number',
    'Reduction',
    'Substitution',
    'Unary',
    'Variable',
    'of_shape',
    'to_funsor',
]
