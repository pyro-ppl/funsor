# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from funsor.domains import Domain, bint, find_domain, make_domain, reals
from funsor.integrate import Integrate
from funsor.interpreter import reinterpret
from funsor.sum_product import MarkovProduct
from funsor.tensor import Tensor, function
from funsor.terms import (
    Cat,
    Funsor,
    Independent,
    Lambda,
    Number,
    Slice,
    Stack,
    Variable,
    of_shape,
    symbolic,
    to_data,
    to_funsor
)
from funsor.util import get_backend, pretty, quote, set_backend

from . import (  # minipyro,  # TODO: enable when minipyro is backend-agnostic
    adjoint,
    affine,
    cnf,
    delta,
    distribution,
    domains,
    einsum,
    gaussian,
    integrate,
    interpreter,
    joint,
    memoize,
    montecarlo,
    ops,
    sum_product,
    terms,
    testing
)

# TODO: move to `funsor.util` when the following circular import issue is resolved
# funsor.domains -> funsor.util -> set_backend -> funsor.torch -> funsor.domains
set_backend(get_backend())


__all__ = [
    'Cat',
    'Domain',
    'Funsor',
    'Independent',
    'Integrate',
    'Lambda',
    'MarkovProduct',
    'Number',
    'Slice',
    'Stack',
    'Tensor',
    'Variable',
    'adjoint',
    'affine',
    'backward',
    'bint',
    'cnf',
    'delta',
    'distribution',
    'domains',
    'einsum',
    'find_domain',
    'function',
    'gaussian',
    'get_backend',
    'integrate',
    'interpreter',
    'joint',
    'make_domain',
    'memoize',
    # 'minipyro',  # TODO: enable when minipyro is backend-agnostic
    'montecarlo',
    'of_shape',
    'ops',
    'pretty',
    'quote',
    'reals',
    'reinterpret',
    'set_backend',
    'sum_product',
    'symbolic',
    'terms',
    'testing',
    'to_data',
    'to_funsor',
]
