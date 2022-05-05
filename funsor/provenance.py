# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import funsor.ops as ops
from funsor.tensor import Tensor
from funsor.terms import Binary, Funsor, FunsorMeta, Number, Unary, Variable, eager


class ProvenanceMeta(FunsorMeta):
    """
    Wrapper to combine provenance information from the term.
    """

    def __call__(cls, term, provenance):
        while isinstance(term, Provenance):
            provenance |= term.provenance
            term = term.term

        return super(ProvenanceMeta, cls).__call__(term, provenance)


class Provenance(Funsor, metaclass=ProvenanceMeta):
    """
    Provenance funsor for tracking the dependence of terms on ``(name, point)``
    of sampled random variables.

    **References**

    [1] David Wingate, Noah Goodman, Andreas Stuhlmüller, Jeffrey Siskind (2011)
        Nonstandard Interpretations of Probabilistic Programs for Efficient Inference
        http://papers.neurips.cc/paper/4309-nonstandard-interpretations-of-probabilistic-programs-for-efficient-inference.pdf

    :param funsor term: A term that depends on tracked variables.
    :param frozenset provenance: A set of tuples of the form ``(name, point)``.
    """

    def __init__(self, term, provenance):
        assert isinstance(term, Funsor)
        assert isinstance(provenance, frozenset)

        provenance_names = frozenset([name for name, point in provenance])
        assert provenance_names.isdisjoint(term.inputs)
        inputs = OrderedDict()
        for name, point in provenance:
            assert isinstance(name, str)
            assert isinstance(point, Funsor)
            assert name not in point.inputs
            inputs.update({name: point.output})
            inputs.update(point.inputs)

        inputs.update(term.inputs)
        output = term.output
        fresh = provenance_names
        bound = {}
        super(Provenance, self).__init__(inputs, output, fresh, bound)
        self.term = term
        self.provenance = provenance

    def eager_subs(self, subs):
        assert isinstance(subs, tuple)
        subs = OrderedDict(subs)
        assert set(subs).issubset(self.fresh)
        new_provenance = frozenset()
        new_term = self.term
        for name, point in self.provenance:
            if name in subs:
                value = subs[name]
                if isinstance(value, Variable):
                    new_provenance |= frozenset([(value.name, point)])
                    continue

                # substituted value needs to match point
                # provenance variable is leaved out
                assert value is point
            else:
                new_provenance |= frozenset([(name, point)])
        return Provenance(new_term, new_provenance) if new_provenance else new_term

    def _sample(self, sampled_vars, sample_inputs, rng_key):
        result = self.term._sample(sampled_vars, sample_inputs, rng_key)
        return Provenance(result, self.provenance)


@eager.register(Binary, ops.BinaryOp, Provenance, Provenance)
def eager_binary_provenance_provenance(op, lhs, rhs):
    return Provenance(op(lhs.term, rhs.term), lhs.provenance | rhs.provenance)


@eager.register(Binary, ops.BinaryOp, Provenance, (Number, Tensor))
def eager_binary_provenance_tensor(op, lhs, rhs):
    assert lhs.fresh.isdisjoint(rhs.inputs)
    return Provenance(op(lhs.term, rhs), lhs.provenance)


@eager.register(Binary, ops.BinaryOp, (Number, Tensor), Provenance)
def eager_binary_tensor_provenance(op, lhs, rhs):
    assert rhs.fresh.isdisjoint(lhs.inputs)
    return Provenance(op(lhs, rhs.term), rhs.provenance)


@eager.register(Unary, ops.UnaryOp, Provenance)
def eager_unary(op, arg):
    return Provenance(op(arg.term), arg.provenance)
