# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import funsor.ops as ops
from funsor.domains import Real
from funsor.terms import Funsor, Unary, Variable, eager
from funsor.util import get_default_dtype


class Sampled(Funsor):
    """
    Funsor that depends on samples in ``terms``.

    :param tuple terms: A tuple of tuples of the form
        ``(name, (point, log_density))``.
    :param funsor arg: A funsor that depends on samples in terms.
    """

    def __init__(self, terms, arg):
        assert isinstance(arg, Funsor)
        assert isinstance(terms, tuple)
        inputs = OrderedDict()
        for name, (point, log_density) in terms:
            assert isinstance(name, str)
            assert isinstance(point, Funsor)
            assert isinstance(log_density, Funsor)
            assert log_density.output == Real
            assert name not in inputs
            assert name not in point.inputs
            inputs.update({name: point.output})
            inputs.update(point.inputs)

        inputs.update(arg.inputs)
        output = arg.output
        fresh = frozenset(name for name, term in terms)
        bound = {}
        super(Sampled, self).__init__(inputs, output, fresh, bound)
        self.arg = arg
        self.terms = terms

    def eager_subs(self, subs):
        assert isinstance(subs, tuple)
        subs = OrderedDict(subs)
        assert set(subs.keys()).issubset(self.fresh)
        new_terms = []
        new_arg = self.arg
        for name, (point, log_density) in self.terms:
            if name in subs:
                value = subs[name]
                assert value.output == point.output
                if isinstance(value, Variable):
                    new_terms.append((value.name, (point, log_density)))
                    continue

                dtype = get_default_dtype()
                is_equal = ops.astype((value == point).all(), dtype)
                new_arg = new_arg * (is_equal.log() + log_density).exp()
            else:
                new_terms.append((name, (point, log_density)))
        return Sampled(tuple(new_terms), new_arg) if new_terms else new_arg

    def _sample(self, sampled_vars, sample_inputs, rng_key):
        return Sampled(
            self.terms, self.arg._sample(sampled_vars, sample_inputs, rng_key)
        )


@eager.register(Unary, ops.UnaryOp, Sampled)
def eager_unary(op, arg):
    return Sampled(arg.terms, op(arg.arg))
