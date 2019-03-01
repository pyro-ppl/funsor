from __future__ import absolute_import, division, print_function

import collections
from six.moves import reduce

from opt_einsum.paths import greedy

from funsor.domains import find_domain
from funsor.interpreter import interpretation, reinterpret
from funsor.ops import ASSOCIATIVE_OPS, DISTRIBUTIVE_OPS
from funsor.registry import KeyedRegistry
from funsor.terms import eager, reflect, Binary, Funsor, Reduce


class Finitary(Funsor):
    """
    Lazy finitary operation. Used internally in the optimizer.
    Finitary(op, operands) == six.moves.reduce(op, operands)
    """
    def __init__(self, op, operands):
        assert callable(op)
        assert isinstance(operands, tuple)
        assert all(isinstance(operand, Funsor) for operand in operands)
        inputs = collections.OrderedDict()
        for operand in operands:
            inputs.update(operand.inputs)

        output = reduce(lambda lhs, rhs: find_domain(op, lhs, rhs),
                        [operand.output for operand in reversed(operands)])

        super(Finitary, self).__init__(inputs, output)
        self.op = op
        self.operands = operands

    def __repr__(self):
        return 'Finitary({}, {})'.format(self.op.__name__, self.operands)

    def eager_subs(self, subs):
        if not any(k in self.inputs for k, v in subs):
            return self
        operands = tuple(operand.eager_subs(subs) for operand in self.operands)
        return Finitary(self.op, operands)


@eager.register(Finitary, object, tuple)
def eager_finitary(op, operands):
    return reduce(lambda lhs, rhs: Binary(op, lhs, rhs), operands)


def associate(cls, *args):
    result = _associate(cls, *args)
    if result is None:
        result = reflect(cls, *args)
    return result


_associate = KeyedRegistry(default=lambda *args: None)
associate.register = _associate.register


@associate.register(Binary, object, Funsor, Funsor)
def binary_to_finitary(op, lhs, rhs):
    """convert Binary to Finitary"""
    return Finitary(op, (lhs, rhs))


@associate.register(Finitary, object, tuple)
def associate_finitary(op, operands):
    # Finitary(Finitary) -> Finitary
    new_operands = []
    for term in operands:
        if isinstance(term, Finitary) and (term.op, op) in ASSOCIATIVE_OPS:
            new_operands.extend(term.operands)
        else:
            new_operands.append(term)

    with interpretation(reflect):
        return Finitary(op, tuple(new_operands))


@associate.register(Reduce, object, Reduce, frozenset)
def associate_reduce(op, arg, reduced_vars):
    """
    Rewrite to the largest possible Reduce(Finitary) by combining Reduces
    Assumes that all input Reduce/Finitary ops have been rewritten
    """
    if (arg.op, op) in ASSOCIATIVE_OPS:
        # Reduce(Reduce) -> Reduce
        new_reduced_vars = reduced_vars.union(arg.reduced_vars)
        return Reduce(op, arg.arg, new_reduced_vars)
    return None


def distribute(cls, *args):
    result = _distribute(cls, *args)
    if result is None:
        result = reflect(cls, *args)
    return result


_distribute = KeyedRegistry(default=lambda *args: None)
distribute.register = _distribute.register


@distribute.register(Finitary, object, tuple)
def distribute_finitary(op, operands):
    if all(isinstance(term, Reduce) for term in operands) and \
            all(term.op == operands[0].op for term in operands) and \
            (operands[0].op, op) in DISTRIBUTIVE_OPS:
        # Finitary(Reduce, Reduce) -> Reduce(Finitary(lhs.arg, rhs.arg))
        new_operands = []
        new_reduced_vars = set()
        for term in operands:
            new_operands.append(term.arg)
            new_reduced_vars = new_reduced_vars.union(term.reduced_vars)

        with interpretation(reflect):
            return Reduce(operands[0].op, Finitary(op, tuple(new_operands)), new_reduced_vars)
    return None


def optimize(cls, *args):
    result = _optimize(cls, *args)
    if result is None:
        result = reflect(cls, *args)
    return result


_optimize = KeyedRegistry(default=lambda *args: None)
optimize.register = _optimize.register


# TODO set a better value for this
REAL_SIZE = 3  # the "size" of a real-valued dimension passed to the path optimizer


@optimize.register(Reduce, object, Finitary, frozenset)
def optimize_reduction(op, arg, reduced_vars):
    r"""
    Recursively convert large Reduce(Finitary) ops to many smaller versions
    by reordering execution with a modified opt_einsum optimizer
    """
    # build opt_einsum optimizer IR
    inputs = []
    size_dict = {}
    for operand in arg.operands:
        inputs.append(frozenset(d for d in operand.inputs.keys()))
        size_dict.update({k: ((REAL_SIZE * v.num_elements) if v.dtype == 'real' else v.dtype)
                          for k, v in operand.inputs.items()})
    outputs = frozenset().union(*inputs) - reduced_vars

    # optimize path with greedy opt_einsum optimizer
    # TODO switch to new 'auto' strategy when it's released
    path = greedy(inputs, outputs, size_dict)

    # convert path IR back to sequence of Reduce(Finitary(...))

    # first prepare a reduce_dim counter to avoid early reduction
    reduce_dim_counter = collections.Counter()
    for input in inputs:
        reduce_dim_counter.update({d: 1 for d in input})

    reduce_op, finitary_op = op, arg.op
    operands = list(arg.operands)
    for (a, b) in path:
        ta = operands[a]
        tb = operands.pop(b)

        # don't reduce a dimension too early - keep a collections.Counter
        # and only reduce when the dimension is removed from all lhs terms in path
        reduce_dim_counter.subtract({d: 1 for d in reduced_vars & frozenset(ta.inputs.keys())})
        reduce_dim_counter.subtract({d: 1 for d in reduced_vars & frozenset(tb.inputs.keys())})

        # reduce variables that don't appear in other terms
        both_vars = frozenset(ta.inputs.keys()) | frozenset(tb.inputs.keys())
        path_end_reduced_vars = frozenset(d for d in reduced_vars & both_vars
                                          if reduce_dim_counter[d] == 0)

        # count new appearance of variables that aren't reduced
        reduce_dim_counter.update({d: 1 for d in reduced_vars & (both_vars - path_end_reduced_vars)})

        with interpretation(reflect):
            path_end = Finitary(finitary_op, (ta, tb))
            if path_end_reduced_vars:
                path_end = Reduce(reduce_op, path_end, path_end_reduced_vars)

        operands[a] = path_end

    # reduce any remaining dims, if necessary
    final_reduced_vars = frozenset(d for (d, count) in reduce_dim_counter.items()
                                   if count > 0) & reduced_vars
    if final_reduced_vars:
        with interpretation(reflect):
            path_end = Reduce(reduce_op, path_end, final_reduced_vars)
    return path_end


def apply_optimizer(x):

    with interpretation(associate):
        x = reinterpret(x)

    with interpretation(distribute):
        x = reinterpret(x)

    with interpretation(optimize):
        x = reinterpret(x)

    return reinterpret(x)  # use previous interpretation
