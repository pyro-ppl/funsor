# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from funsor.domains import Array, Bint, Domain, Real, Reals, bint, find_domain, reals
from funsor.factory import make_funsor
from funsor.integrate import Integrate
from funsor.interpreter import interpretation, reinterpret
from funsor.op_factory import make_op
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
    to_funsor,
)
from funsor.util import get_backend, pretty, quote, set_backend

from . import (  # minipyro,  # TODO: enable when minipyro is backend-agnostic
    adam,
    adjoint,
    affine,
    approximations,
    cnf,
    delta,
    distribution,
    domains,
    einsum,
    elbo,
    gaussian,
    integrate,
    interpretations,
    interpreter,
    inverse_laplace,
    joint,
    montecarlo,
    ops,
    sum_product,
    terms,
    testing,
)

__version__ = "0.4.0"

__all__ = [
    "__version__",
    "Array",
    "Bint",
    "Cat",
    "Domain",
    "Funsor",
    "Independent",
    "Integrate",
    "Lambda",
    "MarkovProduct",
    "Number",
    "Real",
    "Reals",
    "Slice",
    "Stack",
    "Tensor",
    "Variable",
    "adam",
    "adjoint",
    "affine",
    "approximations",
    "backward",
    "bint",
    "cnf",
    "delta",
    "distribution",
    "domains",
    "einsum",
    "elbo",
    "find_domain",
    "function",
    "gaussian",
    "get_backend",
    "integrate",
    "interpretation",
    "interpretations",
    "interpreter",
    "inverse_laplace",
    "joint",
    "make_funsor",
    "make_op",
    "montecarlo",
    "of_shape",
    "ops",
    "pretty",
    "quote",
    "reals",
    "reinterpret",
    "set_backend",
    # 'minipyro',  # TODO: enable when minipyro is backend-agnostic
    "sum_product",
    "symbolic",
    "terms",
    "testing",
    "to_data",
    "to_funsor",
]
