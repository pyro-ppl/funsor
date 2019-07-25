import collections
from functools import reduce

from opt_einsum.paths import greedy

import funsor.ops as ops
from funsor.contract import Contract, contractor
from funsor.delta import MultiDelta
from funsor.domains import find_domain
from funsor.gaussian import Gaussian
from funsor.integrate import Integrate
from funsor.interpreter import dispatched_interpretation, interpretation, reinterpret
from funsor.ops import DISTRIBUTIVE_OPS, UNITS, AssociativeOp
from funsor.terms import Binary, Funsor, Reduce, Unary, eager, lazy, to_funsor
from funsor.torch import Tensor


class Finitary(Funsor):
    """
    Lazy finitary operation. Used internally in the optimizer.
    Finitary(op, operands) == functools.reduce(op, operands)
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


@eager.register(Finitary, AssociativeOp, tuple)
def eager_finitary(op, operands):
    return reduce(op, operands)


@dispatched_interpretation
def associate(cls, *args):
    result = associate.dispatch(cls, *args)
    if result is None:
        result = lazy(cls, *args)
    return result


@associate.register(Binary, AssociativeOp, Funsor, Funsor)
def binary_to_finitary(op, lhs, rhs):
    """convert Binary to Finitary"""
    return Finitary(op, (lhs, rhs))


@associate.register(Finitary, AssociativeOp, tuple)
def associate_finitary(op, operands):
    # Finitary(Finitary) -> Finitary
    new_operands = []
    for term in operands:
        if isinstance(term, Finitary) and term.op is op:
            new_operands.extend(term.operands)
        else:
            new_operands.append(term)

    with interpretation(lazy):
        return Finitary(op, tuple(new_operands))


@associate.register(Reduce, AssociativeOp, Reduce, frozenset)
def associate_reduce(op, arg, reduced_vars):
    """
    Rewrite to the largest possible Reduce(Finitary) by combining Reduces
    Assumes that all input Reduce/Finitary ops have been rewritten
    """
    if arg.op is op:
        # Reduce(Reduce) -> Reduce
        new_reduced_vars = reduced_vars.union(arg.reduced_vars)
        return Reduce(op, arg.arg, new_reduced_vars)
    return None


@dispatched_interpretation
def distribute(cls, *args):
    result = distribute.dispatch(cls, *args)
    if result is None:
        result = lazy(cls, *args)
    return result


@distribute.register(Finitary, AssociativeOp, tuple)
def distribute_finitary(op, operands):
    # TODO raise an error or warning on name collision
    if len(operands) == 1:
        return operands[0]

    reduce_op = None
    reduce_terms, remaining_terms, reduced_vars = [], [], frozenset()
    for term in operands:
        # term is not reduce -> do nothing
        # term is reduce but does not distribute -> do nothing
        # term is reduce and distributes -> put into reduce_terms
        if isinstance(term, Reduce):
            if not reduce_op or not (reduce_op, op) in DISTRIBUTIVE_OPS:
                reduce_op = term.op
            if term.op == reduce_op and (reduce_op, op) in DISTRIBUTIVE_OPS:
                reduce_terms.append(term)
                reduced_vars = reduced_vars | term.reduced_vars
            else:
                remaining_terms.append(term)
        else:
            remaining_terms.append(term)

    if len(reduce_terms) > 1:
        new_finitary_term = Finitary(op, tuple(term.arg for term in reduce_terms))
        remaining_terms.append(Reduce(reduce_op, new_finitary_term, reduced_vars))
        return Finitary(op, tuple(remaining_terms))

    return None


@dispatched_interpretation
def optimize(cls, *args):
    result = optimize.dispatch(cls, *args)
    if result is None:
        result = lazy(cls, *args)
    return result


# TODO set a better value for this
REAL_SIZE = 3  # the "size" of a real-valued dimension passed to the path optimizer


@optimize.register(Reduce, AssociativeOp, Funsor, frozenset)
def optimize_reduction_trivial(op, arg, reduced_vars):
    if not reduced_vars:
        return arg
    return None


@optimize.register(Reduce, AssociativeOp, Binary, frozenset)
@eager.register(Reduce, AssociativeOp, Binary, frozenset)
def optimize_reduce_binary_exp(op, arg, reduced_vars):
    if op is not ops.add or arg.op is not ops.mul or \
            not isinstance(arg.lhs, Unary) or arg.lhs.op is not ops.exp:
        return None
    return Integrate(arg.lhs.arg, arg.rhs, reduced_vars)


@optimize.register(Reduce, AssociativeOp, Finitary, frozenset)
def optimize_reduction(op, arg, reduced_vars):
    r"""
    Recursively convert large Reduce(Finitary) ops to many smaller versions
    by reordering execution with a modified opt_einsum optimizer
    """
    if not reduced_vars:
        return arg

    if not (op, arg.op) in DISTRIBUTIVE_OPS:
        return None

    return Contract(op, arg.op, arg, to_funsor(UNITS[arg.op]), reduced_vars)


@optimize.register(Contract, AssociativeOp, AssociativeOp, Finitary, (Finitary, Funsor, Unary), frozenset)
def optimize_contract_finitary_funsor(sum_op, prod_op, lhs, rhs, reduced_vars):

    if prod_op is not lhs.op:
        return None

    # build opt_einsum optimizer IR
    inputs = [frozenset(t.inputs) for t in lhs.operands] + [frozenset(rhs.inputs)]
    size_dict = {k: ((REAL_SIZE * v.num_elements) if v.dtype == 'real' else v.dtype)
                 for arg in (lhs, rhs) for k, v in arg.inputs.items()}
    outputs = frozenset().union(*inputs) - reduced_vars

    # optimize path with greedy opt_einsum optimizer
    # TODO switch to new 'auto' strategy when it's released
    path = greedy(inputs, outputs, size_dict)

    # convert path IR back to sequence of Reduce(Finitary(...))

    # first prepare a reduce_dim counter to avoid early reduction
    reduce_dim_counter = collections.Counter()
    for input in inputs:
        reduce_dim_counter.update({d: 1 for d in input})

    operands = list(lhs.operands) + [rhs]
    for (a, b) in path:
        b, a = tuple(sorted((a, b), reverse=True))
        tb = operands.pop(b)
        ta = operands.pop(a)

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

        path_end = Contract(sum_op, prod_op, ta, tb, path_end_reduced_vars)
        operands.append(path_end)

    # reduce any remaining dims, if necessary
    final_reduced_vars = frozenset(d for (d, count) in reduce_dim_counter.items()
                                   if count > 0) & reduced_vars
    if final_reduced_vars:
        path_end = Reduce(sum_op, path_end, final_reduced_vars)
    return path_end


@optimize.register(Finitary, AssociativeOp, tuple)
def remove_single_finitary(op, operands):
    if len(operands) == 1:
        return operands[0]
    return None


@optimize.register(Unary, ops.Op, Finitary)
def optimize_exp_finitary(op, arg):
    # useful for handling Integrate...
    if op is not ops.exp or arg.op is not ops.add:
        return None
    return Finitary(ops.mul, tuple(operand.exp() for operand in arg.operands))


@optimize.register(Contract, AssociativeOp, AssociativeOp, Unary, Unary, frozenset)
@optimize.register(Contract, AssociativeOp, AssociativeOp, Funsor, Funsor, frozenset)
@contractor
def optimize_contract(sum_op, prod_op, lhs, rhs, reduced_vars):
    return None


@optimize.register(Contract, AssociativeOp, AssociativeOp, (Unary, Funsor), Finitary, frozenset)
def optimize_contract_funsor_finitary(sum_op, prod_op, lhs, rhs, reduced_vars):
    return Contract(sum_op, prod_op, rhs, lhs, reduced_vars)


@optimize.register(Contract, AssociativeOp, AssociativeOp, Unary, Funsor, frozenset)
def optimize_contract_exp_funsor(sum_op, prod_op, lhs, rhs, reduced_vars):
    if lhs.op is ops.exp and isinstance(lhs.arg, (Gaussian, Tensor, MultiDelta)) and \
            sum_op is ops.add and prod_op is ops.mul:
        return Integrate(lhs.arg, rhs, reduced_vars)
    return None


@optimize.register(Contract, AssociativeOp, AssociativeOp, Funsor, Unary, frozenset)
def optimize_contract_funsor_exp(sum_op, prod_op, lhs, rhs, reduced_vars):
    return Contract(sum_op, prod_op, rhs, lhs, reduced_vars)


@dispatched_interpretation
def desugar(cls, *args):
    result = desugar.dispatch(cls, *args)
    if result is None:
        result = lazy(cls, *args)
    return result


@desugar.register(Finitary, AssociativeOp, tuple)
def desugar_finitary(op, operands):
    return reduce(op, operands)


def apply_optimizer(x):

    with interpretation(associate):
        x = reinterpret(x)

    with interpretation(distribute):
        x = reinterpret(x)

    with interpretation(optimize):
        x = reinterpret(x)

    with interpretation(desugar):
        x = reinterpret(x)

    return reinterpret(x)  # use previous interpretation
