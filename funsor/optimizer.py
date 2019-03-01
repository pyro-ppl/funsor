from __future__ import absolute_import, division, print_function

import collections
from six.moves import reduce

from opt_einsum.paths import greedy

from funsor.domains import find_domain
from funsor.interpreter import interpretation, reinterpret
from funsor.registry import KeyedRegistry
from funsor.terms import eager, reflect, Binary, Funsor, Number, Reduce, Variable
from funsor.torch import Tensor


class Finitary(Funsor):
    """
    Lazy finitary operation. Used internally in the optimizer.
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


def desugar(cls, *args):
    result = _desugar(cls, *args)
    if result is None:
        result = reflect(cls, *args)
    return result


_desugar = KeyedRegistry(default=lambda *args: None)
desugar.register = _desugar.register


@desugar.register(Binary, object, Funsor, Funsor)
def binary_to_finitary(op, lhs, rhs):
    """convert Binary to Finitary"""
    return Finitary(op, (lhs, rhs))


def deoptimize(cls, *args):
    result = _deoptimize(cls, *args)
    if result is None:
        result = reflect(cls, *args)
    return result


_deoptimize = KeyedRegistry(default=lambda *args: None)
deoptimize.register = _deoptimize.register


@deoptimize.register(Finitary, object, tuple)
def deoptimize_finitary(op, operands):
    """
    Rewrite to the largest possible Finitary(Finitary/Reduce) by moving Reduces
    Assumes that all input Finitary ops have been rewritten
    """
    # two cases to rewrite, which we handle in separate branches:
    if all(isinstance(term, (Finitary, Number, Tensor, Variable)) for term in operands):  # TODO check distributivity
        # Case 1) Finitary(Finitary) -> Finitary
        new_operands = []
        for term in operands:
            if isinstance(term, Finitary) and term.op == op:
                new_operands.extend(term.operands)
            else:
                new_operands.append(term)

        with interpretation(reflect):
            return Finitary(op, tuple(new_operands))
    elif all(isinstance(term, Reduce) for term in operands):  # TODO check distributivity
        # Case 2) Finitary(Reduce, Reduce) -> Reduce(Finitary(lhs.arg, rhs.arg))
        new_operands = []
        new_reduced_vars = set()
        for term in operands:
            new_operands.append(term.arg)
            new_reduced_vars = new_reduced_vars.union(term.reduced_vars)

        with interpretation(reflect):
            return Reduce(operands[0].op, Finitary(op, tuple(new_operands)), new_reduced_vars)
    elif all(not isinstance(term, (Reduce, Finitary)) for term in operands):
        with interpretation(reflect):
            return Finitary(op, operands)  # nothing to do, reflect
    else:
        # Note: if we can't rewrite all operands in the finitary, fail for now
        # A more sophisticated strategy is to apply this rule recursively
        # Alternatively, we could do this rewrite on Binary ops instead of Finitary
        raise NotImplementedError("TODO(eb8680) handle mixed case")


@deoptimize.register(Reduce, object, Reduce, frozenset)
def deoptimize_reduce(op, arg, reduced_vars):
    """
    Rewrite to the largest possible Reduce(Finitary) by combining Reduces
    Assumes that all input Reduce/Finitary ops have been rewritten
    """
    if arg.op == op:
        # Reduce(Reduce) -> Reduce
        new_reduced_vars = reduced_vars.union(arg.reduced_vars)
        return Reduce(op, arg.arg, new_reduced_vars)
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
    if not reduced_vars:  # null reduction
        return arg

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
        path_end_finitary = Binary(finitary_op, ta, tb)

        # don't reduce a dimension too early - keep a collections.Counter
        # and only reduce when the dimension is removed from all lhs terms in path
        reduce_dim_counter.subtract({d: 1 for d in reduced_vars & set(ta.inputs.keys())})
        reduce_dim_counter.subtract({d: 1 for d in reduced_vars & set(tb.inputs.keys())})

        # reduce variables that don't appear in other terms
        both_vars = frozenset(ta.inputs.keys()) | frozenset(tb.inputs.keys())
        path_end_reduced_vars = frozenset(d for d in reduced_vars & both_vars
                                          if reduce_dim_counter[d] == 0)

        # count new appearance of variables that aren't reduced
        reduce_dim_counter.update({d: 1 for d in reduced_vars & (both_vars - path_end_reduced_vars)})

        if path_end_reduced_vars:
            path_end = Reduce(reduce_op, path_end_finitary, path_end_reduced_vars)
        else:
            path_end = path_end_finitary

        operands[a] = path_end

    # reduce any remaining dims, if necessary
    final_reduced_vars = frozenset(d for (d, count) in reduce_dim_counter.items()
                                   if count > 0) & reduced_vars
    if final_reduced_vars:
        path_end = Reduce(reduce_op, path_end, final_reduced_vars)
    return path_end


def apply_optimizer(x):

    with interpretation(desugar):
        x = reinterpret(x)

    with interpretation(deoptimize):
        x = reinterpret(x)

    with interpretation(optimize):
        x = reinterpret(x)

    return reinterpret(x)  # use previous interpretation
