import functools
import itertools
import math
import numbers
import re
import typing
from collections import Hashable, OrderedDict
from functools import reduce, singledispatch
from weakref import WeakValueDictionary

from multipledispatch import dispatch
from multipledispatch.variadic import Variadic, isvariadic

import funsor.interpreter as interpreter
import funsor.ops as ops
from funsor.domains import Domain, bint, find_domain, reals
from funsor.interpreter import dispatched_interpretation, interpret
from funsor.ops import AssociativeOp, GetitemOp, Op
from funsor.util import getargspec, lazy_property


def substitute(expr, subs):
    if isinstance(subs, (dict, OrderedDict)):
        subs = tuple(subs.items())
    assert isinstance(subs, tuple)
    assert all(isinstance(v, Funsor) for k, v in subs)

    @interpreter.interpretation(interpreter._INTERPRETATION)  # use base
    def subs_interpreter(cls, *args):
        expr = cls(*args)
        fresh_subs = tuple((k, v) for k, v in subs if k in expr.fresh)
        if fresh_subs:
            expr = interpreter.debug_logged(expr.eager_subs)(fresh_subs)
        return expr

    with interpreter.interpretation(subs_interpreter):
        return interpreter.reinterpret(expr)


def _alpha_mangle(expr):
    """
    Rename bound variables in expr to avoid conflict with any free variables.

    FIXME this does not avoid conflict with other bound variables.
    """
    alpha_subs = {name: interpreter.gensym(name + "__BOUND")
                  for name in expr.bound if "__BOUND" not in name}
    if not alpha_subs:
        return expr

    ast_values = expr._alpha_convert(alpha_subs)
    return reflect(type(expr), *ast_values)


def reflect(cls, *args, **kwargs):
    """
    Construct a funsor, populate ``._ast_values``, and cons hash.
    This is the only interpretation allowed to construct funsors.
    """
    cache_key = tuple(id(arg) if not isinstance(arg, Hashable) else arg for arg in args)
    if cache_key in cls._cons_cache:
        return cls._cons_cache[cache_key]

    arg_types = tuple(typing.Tuple[tuple(map(type, arg))]
                      if (type(arg) is tuple and all(isinstance(a, Funsor) for a in arg))
                      else typing.Tuple if (type(arg) is tuple and not arg)
                      else type(arg) for arg in args)
    cls_specific = (cls.__origin__ if cls.__args__ else cls)[arg_types]
    result = super(FunsorMeta, cls_specific).__call__(*args)
    result._ast_values = args

    # alpha-convert eagerly upon binding any variable
    result = _alpha_mangle(result)

    cls._cons_cache[cache_key] = result
    return result


@dispatched_interpretation
def lazy(cls, *args):
    """
    Substitute eagerly but perform ops lazily.
    """
    result = lazy.dispatch(cls, *args)
    if result is None:
        result = reflect(cls, *args)
    return result


@dispatched_interpretation
def eager(cls, *args):
    """
    Eagerly execute ops with known implementations.
    """
    result = eager.dispatch(cls, *args)
    if result is None:
        result = reflect(cls, *args)
    return result


@dispatched_interpretation
def eager_or_die(cls, *args):
    """
    Eagerly execute ops with known implementations.
    Disallows lazy :class:`Subs` , :class:`Unary` , :class:`Binary` , and
    :class:`Reduce` .

    :raises: :py:class:`NotImplementedError` no pattern is found.
    """
    result = eager.dispatch(cls, *args)
    if result is None:
        if cls in (Subs, Unary, Binary, Reduce):
            raise NotImplementedError("Missing pattern for {}({})".format(
                cls.__name__, ", ".join(map(str, args))))
        result = reflect(cls, *args)
    return result


@dispatched_interpretation
def sequential(cls, *args):
    """
    Eagerly execute ops with known implementations; additonally execute
    vectorized ops sequentially if no known vectorized implementation exists.
    """
    result = sequential.dispatch(cls, *args)
    if result is None:
        result = eager(cls, *args)
    return result


@dispatched_interpretation
def moment_matching(cls, *args):
    """
    A moment matching interpretation of :class:`Reduce` expressions. This falls
    back to :class:`eager` in other cases.
    """
    result = moment_matching.dispatch(cls, *args)
    if result is None:
        result = eager(cls, *args)
    return result


interpreter.set_interpretation(eager)  # Use eager interpretation by default.


class FunsorMeta(type):
    """
    Metaclass for Funsors to perform four independent tasks:

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
    4.  Support subtyping with parameters for pattern matching, e.g. Number[int, int].
    """
    def __init__(cls, name, bases, dct):
        super(FunsorMeta, cls).__init__(name, bases, dct)
        if not hasattr(cls, "__args__"):
            cls.__args__ = ()
        if cls.__args__:
            base, = bases
            cls.__origin__ = base
        else:
            cls._ast_fields = getargspec(cls.__init__)[0][1:]
            cls._cons_cache = WeakValueDictionary()
            cls._type_cache = WeakValueDictionary()

    def __call__(cls, *args, **kwargs):
        if cls.__args__:
            cls = cls.__origin__

        # Convert kwargs to args.
        if kwargs:
            args = list(args)
            for name in cls._ast_fields[len(args):]:
                args.append(kwargs.pop(name))
            assert not kwargs, kwargs
            args = tuple(args)

        return interpret(cls, *args)

    def __getitem__(cls, arg_types):
        if not isinstance(arg_types, tuple):
            arg_types = (arg_types,)
        assert not any(isvariadic(arg_type) for arg_type in arg_types), "nested variadic types not supported"
        # switch tuple to typing.Tuple
        arg_types = tuple(typing.Tuple if arg_type is tuple else arg_type for arg_type in arg_types)
        if arg_types not in cls._type_cache:
            assert not cls.__args__, "cannot subscript a subscripted type {}".format(cls)
            assert len(arg_types) == len(cls._ast_fields), "must provide types for all params"
            new_dct = cls.__dict__.copy()
            new_dct.update({"__args__": arg_types})
            # type(cls) to handle FunsorMeta subclasses
            cls._type_cache[arg_types] = type(cls)(cls.__name__, (cls,), new_dct)
        return cls._type_cache[arg_types]

    def __subclasscheck__(cls, subcls):  # issubclass(subcls, cls)
        if cls is subcls:
            return True
        if not isinstance(subcls, FunsorMeta):
            return super(FunsorMeta, getattr(cls, "__origin__", cls)).__subclasscheck__(subcls)

        cls_origin = getattr(cls, "__origin__", cls)
        subcls_origin = getattr(subcls, "__origin__", subcls)
        if not super(FunsorMeta, cls_origin).__subclasscheck__(subcls_origin):
            return False

        if cls.__args__:
            if not subcls.__args__:
                return False
            if len(cls.__args__) != len(subcls.__args__):
                return False
            for subcls_param, param in zip(subcls.__args__, cls.__args__):
                if not _issubclass_tuple(subcls_param, param):
                    return False
        return True

    @lazy_property
    def classname(cls):
        return cls.__name__ + "[{}]".format(", ".join(
            str(getattr(t, "classname", t))  # Tuple doesn't have __name__
            for t in cls.__args__))


def _issubclass_tuple(subcls, cls):
    """
    utility for pattern matching with tuple subexpressions
    """
    # so much boilerplate...
    cls_is_union = hasattr(cls, "__origin__") and (cls.__origin__ or cls) is typing.Union
    if isinstance(cls, tuple) or cls_is_union:
        return any(_issubclass_tuple(subcls, option)
                   for option in (getattr(cls, "__args__", []) if cls_is_union else cls))

    subcls_is_union = hasattr(subcls, "__origin__") and (subcls.__origin__ or subcls) is typing.Union
    if isinstance(subcls, tuple) or subcls_is_union:
        return any(_issubclass_tuple(option, cls)
                   for option in (getattr(subcls, "__args__", []) if subcls_is_union else subcls))

    subcls_is_tuple = hasattr(subcls, "__origin__") and (subcls.__origin__ or subcls) in (tuple, typing.Tuple)
    cls_is_tuple = hasattr(cls, "__origin__") and (cls.__origin__ or cls) in (tuple, typing.Tuple)
    if subcls_is_tuple != cls_is_tuple:
        return False
    if not cls_is_tuple:
        return issubclass(subcls, cls)
    if not cls.__args__:
        return True
    if not subcls.__args__ or len(subcls.__args__) != len(cls.__args__):
        return False

    return all(_issubclass_tuple(a, b) for a, b in zip(subcls.__args__, cls.__args__))


class Funsor(object, metaclass=FunsorMeta):
    """
    Abstract base class for immutable functional tensors.

    Concrete derived classes must implement ``__init__()`` methods taking
    hashable ``*args`` and no optional ``**kwargs`` so as to support cons
    hashing.

    :param OrderedDict inputs: A mapping from input name to domain.
        This can be viewed as a typed context or a mapping from
        free variables to domains.
    :param Domain output: An output domain.
    """
    def __init__(self, inputs, output, fresh=None, bound=None):
        fresh = frozenset() if fresh is None else fresh
        bound = frozenset() if bound is None else bound
        assert isinstance(inputs, OrderedDict)
        for name, input_ in inputs.items():
            assert isinstance(name, str)
            assert isinstance(input_, Domain)
        assert isinstance(output, Domain)
        assert isinstance(fresh, frozenset)
        assert isinstance(bound, frozenset)
        super(Funsor, self).__init__()
        self.inputs = inputs
        self.output = output
        self.fresh = fresh
        self.bound = bound

    @property
    def dtype(self):
        return self.output.dtype

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, ', '.join(map(repr, self._ast_values)))

    def __str__(self):
        return '{}({})'.format(type(self).__name__, ', '.join(map(str, self._ast_values)))

    def pretty(self, maxlen=40):
        lines = []
        _pretty(self, lines, maxlen)
        return '\n'.join(u'\u2502 ' * indent + text for indent, text in lines)

    def __contains__(self, item):
        raise TypeError

    def _alpha_convert(self, alpha_subs):
        """
        Rename bound variables while preserving all free variables.
        """
        # Substitute all funsor values.
        # Subclasses must handle string conversion.
        assert self.bound.issuperset(alpha_subs)
        return tuple(substitute(v, alpha_subs) for v in self._ast_values)

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
            v = to_funsor(v, self.inputs[k])
            if v.output != self.inputs[k]:
                raise ValueError("Expected substitution of {} to have type {}, but got {}"
                                 .format(repr(k), v.output, self.inputs[k]))
            subs[k] = v
        return Subs(self, tuple(subs.items()))

    def __bool__(self):
        if self.inputs or self.output.shape:
            raise ValueError(
                "bool value of Funsor with more than one value is ambiguous")
        raise NotImplementedError

    def __nonzero__(self):
        return self.__bool__()

    def __len__(self):
        if not self.output.shape:
            raise ValueError('Funsor with empty shape has no len()')
        return self.output.shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def item(self):
        if self.inputs or self.output.shape:
            raise ValueError(
                "only one element Funsors can be converted to Python scalars")
        raise NotImplementedError

    @property
    def requires_grad(self):
        return False

    def reduce(self, op, reduced_vars=None):
        """
        Reduce along all or a subset of inputs.

        :param callable op: A reduction operation.
        :param reduced_vars: An optional input name or set of names to reduce.
            If unspecified, all inputs will be reduced.
        :type reduced_vars: str or frozenset
        """
        assert isinstance(op, AssociativeOp)
        # Eagerly convert reduced_vars to appropriate things.
        if reduced_vars is None:
            # Empty reduced_vars means "reduce over everything".
            reduced_vars = frozenset(self.inputs)
        elif isinstance(reduced_vars, str):
            # A single name means "reduce over this one variable".
            reduced_vars = frozenset([reduced_vars])
        elif isinstance(reduced_vars, set):
            # Support set syntax because it is less verbose.
            reduced_vars = frozenset(reduced_vars)
        assert isinstance(reduced_vars, frozenset), reduced_vars
        if not reduced_vars:
            return self
        assert reduced_vars.issubset(self.inputs)
        return Reduce(op, self, reduced_vars)

    def sample(self, sampled_vars, sample_inputs=None):
        """
        Create a Monte Carlo approximation to this funsor by replacing
        functions of ``sampled_vars`` with :class:`~funsor.delta.Delta` s.

        The result is a :class:`Funsor` with the same ``.inputs`` and
        ``.output`` as the original funsor (plus ``sample_inputs`` if
        provided), so that self can be replaced by the sample in expectation
        computations::

            y = x.sample(sampled_vars)
            assert y.inputs == x.inputs
            assert y.output == x.output
            exact = (x.exp() * integrand).reduce(ops.add)
            approx = (y.exp() * integrand).reduce(ops.add)

        If ``sample_inputs`` is provided, this creates a batch of samples
        scaled samples.

        :param frozenset sampled_vars: A set of input variables to sample.
        :param OrderedDict sample_inputs: An optional mapping from variable
            name to :class:`~funsor.domains.Domain` over which samples will
            be batched.
        """
        assert self.output == reals()
        assert isinstance(sampled_vars, frozenset)
        if sample_inputs is None:
            sample_inputs = OrderedDict()
        assert isinstance(sample_inputs, OrderedDict)
        if sampled_vars.isdisjoint(self.inputs):
            return self

        result = interpreter.debug_logged(self.unscaled_sample)(sampled_vars, sample_inputs)
        if sample_inputs is not None:
            log_scale = 0
            for var, domain in sample_inputs.items():
                if var in result.inputs and var not in self.inputs:
                    log_scale -= math.log(domain.dtype)
            if log_scale != 0:
                result += log_scale
        return result

    def unscaled_sample(self, sampled_vars, sample_inputs):
        """
        Internal method to draw an unscaled sample.
        This should be overridden by subclasses.
        """
        assert self.output == reals()
        assert isinstance(sampled_vars, frozenset)
        assert isinstance(sample_inputs, OrderedDict)
        if sampled_vars.isdisjoint(self.inputs):
            return self
        raise ValueError("Cannot sample from a {}".format(type(self).__name__))

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

    def eager_subs(self, subs):
        """
        Internal substitution function. This relies on the user-facing
        :meth:`__call__` method to coerce non-Funsors to Funsors. Once all
        inputs are Funsors, :meth:`eager_subs` implementations can recurse to
        call :class:`Subs`.
        """
        return None  # defer to default implementation

    def eager_unary(self, op):
        return None  # defer to default implementation

    def eager_reduce(self, op, reduced_vars):
        assert reduced_vars.issubset(self.inputs)  # FIXME Is this valid?
        if not reduced_vars:
            return self

        return None  # defer to default implementation

    def sequential_reduce(self, op, reduced_vars):
        assert reduced_vars.issubset(self.inputs)  # FIXME Is this valid?
        if not reduced_vars:
            return self

        # Try to sum out integer scalars. This is mainly useful for testing,
        # since reduction is more efficiently implemented by Tensor.
        eager_vars = []
        lazy_vars = []
        for k in reduced_vars:
            if isinstance(self.inputs[k].dtype, int) and not self.inputs[k].shape:
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

    def moment_matching_reduce(self, op, reduced_vars):
        assert reduced_vars.issubset(self.inputs)  # FIXME Is this valid?
        if not reduced_vars:
            return self

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

    def sigmoid(self):
        return Unary(ops.sigmoid, self)

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
        if type(other) is not tuple:
            other = to_funsor(other, bint(self.output.shape[0]))
            return Binary(ops.getitem, self, other)

        # Handle Ellipsis slicing.
        if any(part is Ellipsis for part in other):
            left = []
            for part in other:
                if part is Ellipsis:
                    break
                left.append(part)
            right = []
            for part in reversed(other):
                if part is Ellipsis:
                    break
                right.append(part)
            right.reverse()
            missing = len(self.output.shape) - len(left) - len(right)
            assert missing >= 0
            middle = [slice(None)] * missing
            other = tuple(left + middle + right)

        # Handle each slice separately.
        result = self
        offset = 0
        for part in other:
            if isinstance(part, slice):
                if part != slice(None):
                    raise NotImplementedError('TODO support nontrivial slicing')
                offset += 1
            else:
                part = to_funsor(part, bint(result.output.shape[offset]))
                result = Binary(GetitemOp(offset), result, part)
        return result


@singledispatch
def _pretty(arg, lines, maxlen, indent=0):
    line = re.sub('\n\\s*', ' ', str(arg))
    if len(line) > maxlen:
        line = line[:maxlen] + "..."
    lines.append((indent, line))


@_pretty.register(Funsor)
def _(arg, lines, maxlen, indent=0):
    lines.append((indent, type(arg).__name__))
    for arg in arg._ast_values:
        _pretty(arg, lines, maxlen, indent + 1)


@_pretty.register(tuple)
def _(arg, lines, maxlen, indent=0):
    lines.append((indent, type(arg).__name__))
    for item in arg:
        _pretty(item, lines, maxlen, indent + 1)


@_pretty.register(str)
@_pretty.register(Domain)
def _(arg, lines, maxlen, indent=0):
    lines.append((indent, repr(arg)))


interpreter.recursion_reinterpret.register(Funsor)(interpreter.reinterpret_funsor)
interpreter.children.register(Funsor)(interpreter.children_funsor)


@dispatch(object)
def to_funsor(x):
    """
    Convert to a :class:`Funsor`.
    Only :class:`Funsor`s and scalars are accepted.

    :param x: An object.
    :param funsor.domains.Domain output: An optional output hint.
    :return: A Funsor equivalent to ``x``.
    :rtype: Funsor
    :raises: ValueError
    """
    raise ValueError("Cannot convert to Funsor: {}".format(repr(x)))


@dispatch(object, Domain)
def to_funsor(x, output):
    raise ValueError("Cannot convert to Funsor: {}".format(repr(x)))


@dispatch(object, object)
def to_funsor(x, output):
    raise TypeError("Invalid Domain: {}".format(repr(output)))


@dispatch(Funsor)
def to_funsor(x):
    return x


@dispatch(Funsor, Domain)
def to_funsor(x, output):
    if x.output != output:
        raise ValueError("Output mismatch: {} vs {}".format(x.output, output))
    return x


@singledispatch
def to_data(x):
    """
    Extract a python object from a :class:`Funsor`.

    Raises a ``ValueError`` if free variables remain or if the funsor is lazy.

    :param x: An object, possibly a :class:`Funsor`.
    :return: A non-funsor equivalent to ``x``.
    :raises: ValueError
    """
    return x


@to_data.register(Funsor)
def _to_data_funsor(x):
    raise ValueError("cannot convert to a non-Funsor: {}".format(repr(x)))


class Variable(Funsor):
    """
    Funsor representing a single free variable.

    :param str name: A variable name.
    :param funsor.domains.Domain output: A domain.
    """
    def __init__(self, name, output):
        inputs = OrderedDict([(name, output)])
        fresh = frozenset({name})
        super(Variable, self).__init__(inputs, output, fresh)
        self.name = name

    def __repr__(self):
        return "Variable({}, {})".format(repr(self.name), repr(self.output))

    def __str__(self):
        return self.name

    def eager_subs(self, subs):
        assert len(subs) == 1 and subs[0][0] == self.name
        return subs[0][1]


@dispatch(str, Domain)
def to_funsor(name, output):
    return Variable(name, output)


class Subs(Funsor):
    """
    Lazy substitution of the form ``x(u=y, v=z)``.
    """
    def __init__(self, arg, subs):
        assert isinstance(arg, Funsor)
        assert isinstance(subs, tuple)
        for key, value in subs:
            assert isinstance(key, str)
            assert key in arg.inputs
            assert isinstance(value, Funsor)
        inputs = arg.inputs.copy()
        for key, value in subs:
            del inputs[key]
        for key, value in subs:
            inputs.update(value.inputs)
        fresh = frozenset()
        bound = frozenset(key for key, value in subs if key not in inputs)
        super(Subs, self).__init__(inputs, arg.output, fresh, bound)
        self.arg = arg
        self.subs = OrderedDict(subs)

    def __repr__(self):
        return 'Subs({}, {})'.format(self.arg, self.subs)

    def _alpha_convert(self, alpha_subs):
        alpha_subs = {k: to_funsor(v, self.subs[k].output)
                      for k, v in alpha_subs.items()}
        arg, subs = super()._alpha_convert(alpha_subs)
        subs = tuple((str(alpha_subs.get(k, k)), v) for k, v in subs)
        return arg, subs

    def unscaled_sample(self, sampled_vars, sample_inputs):
        if any(k in sample_inputs for k, v in self.subs.items()):
            raise NotImplementedError('TODO alpha-convert')
        subs_sampled_vars = set()
        for name in sampled_vars:
            if name in self.arg.inputs:
                if any(name in v.inputs for k, v in self.subs.items()):
                    raise ValueError("Cannot sample")
                subs_sampled_vars.add(name)
            else:
                for k, v in self.subs.items():
                    if name in v.inputs:
                        subs_sampled_vars.add(k)
        subs_sampled_vars = frozenset(subs_sampled_vars)
        arg = self.arg.unscaled_sample(subs_sampled_vars, sample_inputs)
        return Subs(arg, tuple(self.subs.items()))


@lazy.register(Subs, Funsor, object)
@eager.register(Subs, Funsor, object)
def eager_subs(arg, subs):
    assert isinstance(subs, tuple)
    if not any(k in arg.inputs for k, v in subs):
        return arg
    return substitute(arg, subs)


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


@eager.register(Unary, Op, Funsor)
def eager_unary(op, arg):
    return interpreter.debug_logged(arg.eager_unary)(op)


@eager.register(Unary, AssociativeOp, Funsor)
def eager_unary(op, arg):
    if not arg.output.shape:
        return arg
    return interpreter.debug_logged(arg.eager_unary)(op)


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

    def eager_reduce(self, op, reduced_vars):
        if op is self.op:
            lhs = self.lhs.reduce(op, reduced_vars)
            rhs = self.rhs.reduce(op, reduced_vars)
            return op(lhs, rhs)
        return interpreter.debug_logged(super(Binary, self).eager_reduce)(op, reduced_vars)

    def unscaled_sample(self, sampled_vars, sample_inputs=None):
        if self.op is ops.logaddexp:
            # Sample mixture components independently.
            lhs = self.lhs.unscaled_sample(sampled_vars, sample_inputs)
            rhs = self.rhs.unscaled_sample(sampled_vars, sample_inputs)
            return Binary(ops.logaddexp, lhs, rhs)
        raise TypeError("Cannot sample from Binary({}, ...)".format(self.op))


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
        fresh = frozenset()
        bound = reduced_vars
        super(Reduce, self).__init__(inputs, output, fresh, bound)
        self.op = op
        self.arg = arg
        self.reduced_vars = reduced_vars

    def __repr__(self):
        return 'Reduce({}, {}, {})'.format(
            self.op.__name__, self.arg, self.reduced_vars)

    def _alpha_convert(self, alpha_subs):
        alpha_subs = {k: to_funsor(v, self.arg.inputs[k])
                      for k, v in alpha_subs.items()}
        op, arg, reduced_vars = super()._alpha_convert(alpha_subs)
        reduced_vars = frozenset(str(alpha_subs.get(k, k)) for k in reduced_vars)
        return op, arg, reduced_vars

    def eager_reduce(self, op, reduced_vars):
        if op is self.op:
            # Eagerly fuse reductions.
            assert isinstance(reduced_vars, frozenset)
            reduced_vars = reduced_vars.intersection(self.inputs) | self.reduced_vars
            return Reduce(op, self.arg, reduced_vars)
        return super(Reduce, self).eager_reduce(op, reduced_vars)

    def unscaled_sample(self, sampled_vars, sample_inputs=None):
        if self.op is ops.logaddexp:
            arg = self.arg.unscaled_sample(sampled_vars, sample_inputs)
            return Reduce(ops.logaddexp, arg, self.reduced_vars)
        raise TypeError("Cannot sample from Reduce({}, ...)".format(self.op))


@eager.register(Reduce, AssociativeOp, Funsor, frozenset)
def eager_reduce(op, arg, reduced_vars):
    return interpreter.debug_logged(arg.eager_reduce)(op, reduced_vars)


@eager.register(Binary, AssociativeOp, Reduce, (Funsor, Reduce))
def eager_distribute_reduce_other(op, red, other):
    if (red.op, op) in ops.DISTRIBUTIVE_OPS:
        # Use distributive law.
        arg = op(red.arg, other)
        return arg.reduce(red.op, red.reduced_vars)

    return None  # defer to default implementation


@eager.register(Binary, AssociativeOp, Funsor, Reduce)
def eager_distribute_other_reduce(op, other, red):
    if (red.op, op) in ops.DISTRIBUTIVE_OPS:
        # Use distributive law.
        arg = op(other, red.arg)
        return arg.reduce(red.op, red.reduced_vars)

    return None  # defer to default implementation


@sequential.register(Reduce, AssociativeOp, Funsor, frozenset)
def sequential_reduce(op, arg, reduced_vars):
    return interpreter.debug_logged(arg.sequential_reduce)(op, reduced_vars)


@moment_matching.register(Reduce, AssociativeOp, Funsor, frozenset)
def moment_matching_reduce(op, arg, reduced_vars):
    return interpreter.debug_logged(arg.moment_matching_reduce)(op, reduced_vars)


class NumberMeta(FunsorMeta):
    """
    Wrapper to fill in default ``dtype``.
    """
    def __call__(cls, data, dtype=None):
        if dtype is None:
            dtype = "real"
        return super(NumberMeta, cls).__call__(data, dtype)


class Number(Funsor, metaclass=NumberMeta):
    """
    Funsor backed by a Python number.

    :param numbers.Number data: A python number.
    :param dtype: A nonnegative integer or the string "real".
    """
    def __init__(self, data, dtype=None):
        assert isinstance(data, numbers.Number)
        if isinstance(dtype, int):
            data = type(dtype)(data)
            if dtype != 2:  # booleans have bitwise interpretation
                assert 0 <= data and data < dtype
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

    def eager_unary(self, op):
        dtype = find_domain(op, self.output).dtype
        return Number(op(self.data), dtype)


@dispatch(numbers.Number)
def to_funsor(x):
    return Number(x)


@dispatch(numbers.Number, Domain)
def to_funsor(x, output):
    if output.shape:
        raise ValueError("Cannot create Number with shape {}".format(output.shape))
    return Number(x, output.dtype)


@to_data.register(Number)
def _to_data_number(x):
    return x.data


@eager.register(Binary, Op, Number, Number)
def eager_binary_number_number(op, lhs, rhs):
    data = op(lhs.data, rhs.data)
    output = find_domain(op, lhs.output, rhs.output)
    dtype = output.dtype
    return Number(data, dtype)


class SliceMeta(FunsorMeta):
    """
    Wrapper to fill in ``start``, ``stop``, ``step``, ``dtype`` following
    Python conventions.
    """
    def __call__(cls, name, *args, **kwargs):
        start = 0
        step = 1
        dtype = None
        if len(args) == 1:
            stop = args[0]
            dtype = kwargs.pop("dtype", stop)
        elif len(args) == 2:
            start, stop = args
            dtype = kwargs.pop("dtype", stop)
        elif len(args) == 3:
            start, stop, step = args
            dtype = kwargs.pop("dtype", stop)
        elif len(args) == 4:
            start, stop, step, dtype = args
        else:
            raise ValueError
        if step <= 0:
            raise ValueError
        return super().__call__(name, start, stop, step, dtype)


class Slice(Funsor, metaclass=SliceMeta):
    """
    Symbolic representation of a Python :py:class:`slice` object.
    """
    def __init__(self, name, start, stop, step, dtype):
        assert isinstance(name, str)
        assert start is None or isinstance(start, int)
        assert stop is None or isinstance(stop, int)
        assert step is None or isinstance(step, int)
        assert isinstance(dtype, int)
        size = max(0, (stop + step - 1 - start) // step)
        inputs = OrderedDict([(name, bint(size))])
        output = bint(dtype)
        fresh = frozenset({"name"})
        super().__init__(inputs, output, fresh)
        self.name = name
        self.slice = slice(start, stop, step)


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
        fresh = frozenset()  # TODO get this right
        bound = frozenset()
        super(Align, self).__init__(inputs, output, fresh, bound)
        self.arg = arg

    def align(self, names):
        return self.arg.align(names)

    def eager_unary(self, op):
        return Unary(op, self.arg)

    def eager_reduce(self, op, reduced_vars):
        return self.arg.reduce(op, reduced_vars)


@eager.register(Align, Funsor, tuple)
def eager_align(arg, names):
    if not frozenset(names) == frozenset(arg.inputs.keys()):
        # assume there's been a substitution and this align is no longer valid
        return arg
    return None


@eager.register(Binary, Op, Align, Funsor)
def eager_binary_align_funsor(op, lhs, rhs):
    return Binary(op, lhs.arg, rhs)


@eager.register(Binary, Op, Funsor, Align)
def eager_binary_funsor_align(op, lhs, rhs):
    return Binary(op, lhs, rhs.arg)


@eager.register(Binary, Op, Align, Align)
def eager_binary_align_align(op, lhs, rhs):
    return Binary(op, lhs.arg, rhs.arg)


eager.register(Binary, AssociativeOp, Reduce, Align)(eager_distribute_reduce_other)
eager.register(Binary, AssociativeOp, Align, Reduce)(eager_distribute_other_reduce)


class Stack(Funsor):
    """
    Stack of funsors along a new input dimension.

    :param str name: The name of the new input variable along which to stack.
    :param tuple parts: A tuple of Funsors of homogenous output domain.
    """
    def __init__(self, name, parts):
        assert isinstance(name, str)
        assert isinstance(parts, tuple)
        assert parts
        assert not any(name in x.inputs for x in parts)
        assert len(set(x.output for x in parts)) == 1
        output = parts[0].output
        domain = bint(len(parts))
        inputs = OrderedDict([(name, domain)])
        for x in parts:
            inputs.update(x.inputs)
        fresh = frozenset({name})
        super().__init__(inputs, output, fresh)
        self.name = name
        self.parts = parts

    def eager_subs(self, subs):
        assert isinstance(subs, tuple) and len(subs) == 1 and subs[0][0] == self.name
        index = subs[0][1]

        # Try to eagerly select an index.
        assert index.output == bint(len(self.parts))

        if isinstance(index, Number):
            # Select a single part.
            return self.parts[index.data]
        elif isinstance(index, Variable):
            # Rename the stacking dimension.
            parts = self.parts
            return Stack(index.name, parts)
        elif isinstance(index, Slice):
            parts = self.parts[index.slice]
            return Stack(index.name, parts)
        else:
            raise NotImplementedError('TODO support advanced indexing in Stack')

    def eager_reduce(self, op, reduced_vars):
        parts = self.parts
        if self.name in reduced_vars:
            reduced_vars -= frozenset([self.name])
            if reduced_vars:
                parts = tuple(x.reduce(op, reduced_vars) for x in parts)
            return reduce(op, parts)
        parts = tuple(x.reduce(op, reduced_vars) for x in parts)
        return Stack(self.name, parts)


@eager.register(Stack, str, tuple)
def eager_stack(name, parts):
    return eager_stack_homogeneous(name, *parts)


@dispatch(str, Variadic[Funsor])
def eager_stack_homogeneous(name, *parts):
    return None  # defer to default implementation


class Cat(Funsor):
    """
    Concatenate funsors along an existing input dimension.

    :param str name: The name of the input variable along which to concatenate.
    :param tuple parts: A tuple of Funsors of homogenous output domain.
    """
    def __init__(self, name, parts):
        assert isinstance(name, str)
        assert isinstance(parts, tuple)
        assert parts
        assert all(name in x.inputs for x in parts)
        assert len(set(x.output for x in parts)) == 1
        output = parts[0].output
        inputs = OrderedDict()
        for x in parts:
            inputs.update(x.inputs)
        inputs[name] = bint(sum(x.inputs[name].size for x in parts))
        super().__init__(inputs, output)
        self.name = name
        self.parts = parts


@eager.register(Cat, str, tuple)
def eager_cat(name, parts):
    if len(parts) == 1:
        return parts[0]
    return eager_cat_homogeneous(name, *parts)


@dispatch(str, Variadic[Funsor])
def eager_cat_homogeneous(name, *parts):
    return None  # defer to default implementation


@dispatch(str, Variadic[Stack])
def eager_cat_homogeneous(name, *parts):
    if all(p.name == name for p in parts):
        parts = sum((p.parts for p in parts), ())
        return Stack(name, parts)

    return None  # defer to default implementation


class Lambda(Funsor):
    """
    Lazy inverse to ``ops.getitem``.

    This is useful to simulate higher-order functions of integers
    by representing those functions as arrays.
    """
    def __init__(self, var, expr):
        assert isinstance(var, Variable)
        assert isinstance(var.dtype, int)
        assert isinstance(expr, Funsor)
        inputs = expr.inputs.copy()
        inputs.pop(var.name, None)
        shape = (var.dtype,) + expr.output.shape
        output = Domain(shape, expr.dtype)
        fresh = frozenset()
        bound = frozenset({var.name})  # TODO make sure this is correct
        super(Lambda, self).__init__(inputs, output, fresh, bound)
        self.var = var
        self.expr = expr

    def _alpha_convert(self, alpha_subs):
        alpha_subs = {k: to_funsor(v, self.var.inputs[k])
                      for k, v in alpha_subs.items()}
        return super()._alpha_convert(alpha_subs)


@eager.register(Binary, GetitemOp, Lambda, (Funsor, Align))
def eager_getitem_lambda(op, lhs, rhs):
    if op.offset == 0:
        return Subs(lhs.expr, ((lhs.var.name, rhs),))
    expr = GetitemOp(op.offset - 1)(lhs.expr, rhs)
    return Lambda(lhs.var, expr)


class Independent(Funsor):
    """
    Creates an independent diagonal distribution.

    This is equivalent to substitution followed by reduction::

        f = ...  # a batched distribution
        assert f.inputs['x_i'] == reals(4, 5)
        assert f.inputs['i'] == bint(3)

        g = Independent(f, 'x', 'i', 'x_i')
        assert g.inputs['x'] == reals(3, 4, 5)
        assert 'x_i' not in g.inputs
        assert 'i' not in g.inputs

        x = Variable('x', reals(3, 4, 5))
        g == f(x_i=x['i']).reduce(ops.logaddexp, 'i')
    """
    def __init__(self, fn, reals_var, bint_var, diag_var):
        assert isinstance(fn, Funsor)
        assert isinstance(reals_var, str)
        assert isinstance(bint_var, str)
        assert bint_var in fn.inputs
        assert isinstance(fn.inputs[bint_var].dtype, int)
        assert isinstance(diag_var, str)
        assert diag_var in fn.inputs
        assert fn.inputs[diag_var].dtype == 'real'
        inputs = fn.inputs.copy()
        shape = (inputs.pop(bint_var).dtype,) + inputs.pop(diag_var).shape
        assert reals_var not in inputs
        inputs[reals_var] = reals(*shape)
        fresh = frozenset({reals_var})
        bound = frozenset({bint_var, diag_var})
        super(Independent, self).__init__(inputs, fn.output, fresh, bound)
        self.fn = fn
        self.reals_var = reals_var
        self.bint_var = bint_var
        self.diag_var = diag_var

    def _alpha_convert(self, alpha_subs):
        alpha_subs = {k: to_funsor(v, self.fn.inputs[k])
                      for k, v in alpha_subs.items()}
        fn, reals_var, bint_var, diag_var = super()._alpha_convert(alpha_subs)
        bint_var = str(alpha_subs.get(bint_var, bint_var))
        diag_var = str(alpha_subs.get(diag_var, diag_var))
        return fn, reals_var, bint_var, diag_var

    def unscaled_sample(self, sampled_vars, sample_inputs):
        if self.bint_var in sampled_vars or self.bint_var in sample_inputs:
            raise NotImplementedError('TODO alpha-convert')
        sampled_vars = frozenset(self.diag_var if v == self.reals_var else v
                                 for v in sampled_vars)
        fn = self.fn.unscaled_sample(sampled_vars, sample_inputs)
        return Independent(fn, self.reals_var, self.bint_var, self.diag_var)

    def eager_subs(self, subs):
        assert len(subs) == 1 and subs[0][0] == self.reals_var
        value = subs[0][1]

        # Handle simple renaming to preserve Independent.
        if isinstance(value, Variable):
            return Independent(self.fn, value.name, self.bint_var, self.diag_var)

        # Otherwise convert to a Reduce.
        result = Subs(self.fn, ((self.diag_var, value[self.bint_var]),))
        result = result.reduce(ops.add, self.bint_var)
        return result


@eager.register(Independent, Funsor, str, str, str)
def eager_independent_trivial(fn, reals_var, bint_var, diag_var):
    # compare to Independent.eager_subs
    if diag_var not in fn.inputs:
        return fn.reduce(ops.add, bint_var)
    return None


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


################################################################################
# Register Ops
################################################################################

@ops.abs.register(Funsor)
def _abs(x):
    return Unary(ops.abs, x)


@ops.sqrt.register(Funsor)
def _sqrt(x):
    return Unary(ops.sqrt, x)


@ops.exp.register(Funsor)
def _exp(x):
    return Unary(ops.exp, x)


@ops.log.register(Funsor)
def _log(x):
    return Unary(ops.log, x)


@ops.log1p.register(Funsor)
def _log1p(x):
    return Unary(ops.log1p, x)


@ops.sigmoid.register(Funsor)
def _sigmoid(x):
    return Unary(ops.sigmoid, x)


__all__ = [
    'Binary',
    'Cat',
    'Funsor',
    'Independent',
    'Lambda',
    'Number',
    'Reduce',
    'Stack',
    'Slice',
    'Subs',
    'Unary',
    'Variable',
    'eager',
    'eager_or_die',
    'lazy',
    'moment_matching',
    'of_shape',
    'reflect',
    'sequential',
    'to_data',
    'to_funsor',
]
