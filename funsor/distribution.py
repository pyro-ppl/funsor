# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
import importlib
import inspect
import math
import typing
import warnings
from collections import OrderedDict
from importlib import import_module

import makefun

import funsor.delta
import funsor.ops as ops
from funsor.affine import is_affine
from funsor.cnf import Contraction, GaussianMixture
from funsor.domains import Array, Real, Reals
from funsor.gaussian import Gaussian
from funsor.interpreter import gensym
from funsor.tensor import (Tensor, align_tensors, dummy_numeric_array, get_default_prototype,
                           ignore_jit_warnings, numeric_array, stack)
from funsor.terms import Funsor, FunsorMeta, Independent, Number, Variable, \
    eager, reflect, to_data, to_funsor
from funsor.util import broadcast_shape, get_backend, getargspec, lazy_property


BACKEND_TO_DISTRIBUTIONS_BACKEND = {
    "torch": "funsor.torch.distributions",
    "jax": "funsor.jax.distributions",
}


def numbers_to_tensors(*args):
    """
    Convert :class:`~funsor.terms.Number` s to :class:`funsor.tensor.Tensor` s,
    using any provided tensor as a prototype, if available.
    """
    if any(isinstance(x, Number) for x in args):
        prototype = get_default_prototype()
        options = dict(dtype=prototype.dtype)
        for x in args:
            if isinstance(x, Tensor):
                options = dict(dtype=x.data.dtype, device=getattr(x.data, "device", None))
                break
        with ignore_jit_warnings():
            args = tuple(Tensor(numeric_array(x.data, **options), dtype=x.dtype)
                         if isinstance(x, Number) else x
                         for x in args)
    return args


class DistributionMeta(FunsorMeta):
    """
    Wrapper to fill in default values and convert Numbers to Tensors.
    """
    def __call__(cls, *args, **kwargs):
        kwargs.update(zip(cls._ast_fields, args))
        kwargs["value"] = kwargs.get("value", "value")
        kwargs = OrderedDict((k, kwargs[k]) for k in cls._ast_fields)  # make sure args are sorted

        domains = OrderedDict()
        for k, v in kwargs.items():
            if k == "value":
                continue

            # compute unbroadcasted param domains
            domain = cls._infer_param_domain(k, getattr(kwargs[k], "shape", ()))
            # use to_funsor to infer output dimensions of e.g. tensors
            domains[k] = domain if domain is not None else to_funsor(v).output

            # broadcast individual param domains with Funsor inputs
            # this avoids .expand-ing underlying parameter tensors
            dtype = domains[k].dtype
            if isinstance(v, Funsor):
                domains[k] = Array[dtype, broadcast_shape(v.shape, domains[k].shape)]
            elif ops.is_numeric_array(v):
                domains[k] = Array[dtype, broadcast_shape(v.shape, domains[k].shape)]

        # now use the broadcasted parameter shapes to infer the event_shape
        domains["value"] = cls._infer_value_domain(**domains)

        # finally, perform conversions to funsors
        kwargs = OrderedDict((k, to_funsor(v, output=domains[k])) for k, v in kwargs.items())
        args = numbers_to_tensors(*kwargs.values())

        return super(DistributionMeta, cls).__call__(*args)


class Distribution(Funsor, metaclass=DistributionMeta):
    r"""
    Funsor backed by a PyTorch/JAX distribution object.

    :param \*args: Distribution-dependent parameters.  These can be either
        funsors or objects that can be coerced to funsors via
        :func:`~funsor.terms.to_funsor` . See derived classes for details.
    """
    dist_class = "defined by derived classes"

    def __init__(self, *args):
        params = tuple(zip(self._ast_fields, args))
        assert any(k == 'value' for k, v in params)
        inputs = OrderedDict()
        for name, value in params:
            assert isinstance(name, str)
            assert isinstance(value, Funsor)
            inputs.update(value.inputs)
        inputs = OrderedDict(inputs)
        output = Real
        super(Distribution, self).__init__(inputs, output)
        self.params = OrderedDict(params)

    def __repr__(self):
        return '{}({})'.format(type(self).__name__,
                               ', '.join('{}={}'.format(*kv) for kv in self.params.items()))

    def eager_reduce(self, op, reduced_vars):
        assert reduced_vars.issubset(self.inputs)
        if op is ops.logaddexp and isinstance(self.value, Variable) and self.value.name in reduced_vars:
            return Number(0.)  # distributions are normalized
        return super(Distribution, self).eager_reduce(op, reduced_vars)

    def _get_raw_dist(self):
        """
        Internal method for working with underlying distribution attributes
        """
        value_name = [name for name, domain in self.value.inputs.items()  # TODO is this right?
                      if domain == self.value.output][0]
        # arbitrary name-dim mapping, since we're converting back to a funsor anyway
        name_to_dim = {name: -dim-1 for dim, (name, domain) in enumerate(self.inputs.items())
                       if isinstance(domain.dtype, int) and name != value_name}
        raw_dist = to_data(self, name_to_dim=name_to_dim)
        dim_to_name = {dim: name for name, dim in name_to_dim.items()}
        # also return value output, dim_to_name for converting results back to funsor
        value_output = self.inputs[value_name]
        return raw_dist, value_name, value_output, dim_to_name

    @property
    def has_enumerate_support(self):
        return getattr(self.dist_class, "has_enumerate_support", False)

    @classmethod
    def eager_log_prob(cls, *params):
        params, value = params[:-1], params[-1]
        params = params + (Variable("value", value.output),)
        instance = reflect(cls, *params)
        raw_dist, value_name, value_output, dim_to_name = instance._get_raw_dist()
        assert value.output == value_output
        name_to_dim = {v: k for k, v in dim_to_name.items()}
        dim_to_name.update({-1 - d - len(raw_dist.batch_shape): name
                            for d, name in enumerate(value.inputs) if name not in name_to_dim})
        name_to_dim.update({v: k for k, v in dim_to_name.items() if v not in name_to_dim})
        raw_log_prob = raw_dist.log_prob(to_data(value, name_to_dim=name_to_dim))
        log_prob = to_funsor(raw_log_prob, Real, dim_to_name=dim_to_name)
        # this logic ensures that the inputs have the canonical order
        # implied by align_tensors, which is assumed pervasively in tests
        inputs = OrderedDict()
        for x in params[:-1] + (value,):
            inputs.update(x.inputs)
        return log_prob.align(tuple(inputs))

    def unscaled_sample(self, sampled_vars, sample_inputs, rng_key=None):

        # note this should handle transforms correctly via distribution_to_data
        raw_dist, value_name, value_output, dim_to_name = self._get_raw_dist()
        for d, name in zip(range(len(sample_inputs), 0, -1), sample_inputs.keys()):
            dim_to_name[-d - len(raw_dist.batch_shape)] = name

        if value_name not in sampled_vars:
            return self

        sample_shape = tuple(v.size for v in sample_inputs.values())
        sample_args = (sample_shape,) if get_backend() == "torch" else (rng_key, sample_shape)
        if raw_dist.has_rsample:
            raw_value = raw_dist.rsample(*sample_args)
        else:
            raw_value = ops.detach(raw_dist.sample(*sample_args))

        funsor_value = to_funsor(raw_value, output=value_output, dim_to_name=dim_to_name)
        funsor_value = funsor_value.align(
            tuple(sample_inputs) + tuple(inp for inp in self.inputs if inp in funsor_value.inputs))
        result = funsor.delta.Delta(value_name, funsor_value)
        if not raw_dist.has_rsample:
            # scaling of dice_factor by num samples should already be handled by Funsor.sample
            raw_log_prob = raw_dist.log_prob(raw_value)
            dice_factor = to_funsor(raw_log_prob - ops.detach(raw_log_prob),
                                    output=self.output, dim_to_name=dim_to_name)
            result = result + dice_factor
        return result

    def enumerate_support(self, expand=False):
        assert self.has_enumerate_support and isinstance(self.value, Variable)
        raw_dist, value_name, value_output, dim_to_name = self._get_raw_dist()
        raw_value = raw_dist.enumerate_support(expand=expand)
        dim_to_name[min(dim_to_name.keys(), default=0)-1] = value_name
        return to_funsor(raw_value, output=value_output, dim_to_name=dim_to_name)

    def entropy(self):
        raw_dist, value_name, value_output, dim_to_name = self._get_raw_dist()
        raw_value = raw_dist.entropy()
        return to_funsor(raw_value, output=self.output, dim_to_name=dim_to_name)

    def mean(self):
        raw_dist, value_name, value_output, dim_to_name = self._get_raw_dist()
        raw_value = raw_dist.mean
        return to_funsor(raw_value, output=value_output, dim_to_name=dim_to_name)

    def variance(self):
        raw_dist, value_name, value_output, dim_to_name = self._get_raw_dist()
        raw_value = raw_dist.variance
        return to_funsor(raw_value, output=value_output, dim_to_name=dim_to_name)

    def __getattribute__(self, attr):
        if attr in type(self)._ast_fields and attr != 'name':
            return self.params[attr]
        return super().__getattribute__(attr)

    @classmethod
    def _infer_value_dtype(cls, domains):
        try:
            support = cls.dist_class.support
        except NotImplementedError:
            raise NotImplementedError(f"Failed to infer dtype of {cls.dist_class.__name__}")
        while type(support).__name__ == "IndependentConstraint":
            support = support.base_constraint
        if type(support).__name__ == "_IntegerInterval":
            return int(support.upper_bound + 1)
        return "real"

    @classmethod
    @functools.lru_cache(maxsize=5000)
    def _infer_value_domain(cls, **domains):
        dtype = cls._infer_value_dtype(domains)
        # TODO implement .infer_shapes() methods on each distribution
        # TODO fix distribution constraints by wrapping in _Independent
        batch_shape, event_shape = infer_shapes(cls.dist_class, domains)
        shape = batch_shape + event_shape
        if "value" in domains:
            shape = broadcast_shape(shape, domains["value"].shape)
        return Array[dtype, shape]

    @classmethod
    @functools.lru_cache(maxsize=5000)
    def _infer_param_domain(cls, name, raw_shape):
        support = cls.dist_class.arg_constraints.get(name, None)
        # XXX: if the backend does not have the same definition of constraints, we should
        # define backend-specific distributions and overide these `infer_value_domain`,
        # `infer_param_domain` methods.
        # Because NumPyro and Pyro have the same pattern, we use name check for simplicity.
        support_name = type(support).__name__.lstrip("_")

        event_dim = 0
        while support_name == "IndependentConstraint":
            event_dim += support.reinterpreted_batch_ndims
            support = support.base_constraint
            support_name = type(support).__name__.lstrip("_")

        if support_name == "Simplex":
            output = Reals[raw_shape[-1 - event_dim:]]
        elif support_name == "RealVector":
            output = Reals[raw_shape[-1 - event_dim:]]
        elif support_name in ["LowerCholesky", "PositiveDefinite"]:
            output = Reals[raw_shape[-2 - event_dim:]]
        # resolve the issue: logits's constraints are real (instead of real_vector)
        # for discrete multivariate distributions in Pyro
        elif support_name == "Real":
            if name == "logits" and (
                    "probs" in cls.dist_class.arg_constraints
                    and type(cls.dist_class.arg_constraints["probs"]).__name__.lstrip("_") == "Simplex"):
                output = Reals[raw_shape[-1 - event_dim:]]
            else:
                output = Reals[raw_shape[len(raw_shape) - event_dim:]]
        elif support_name in ("Interval", "GreaterThan", "LessThan"):
            output = Reals[raw_shape[len(raw_shape) - event_dim:]]
        else:
            output = None
        return output


def infer_shapes(dist_class, domains):
    arg_shapes = {k: domain.shape for k, domain in domains.items() if k != "value"}
    try:
        return dist_class.infer_shapes(**arg_shapes)
    except (AttributeError, NotImplementedError):
        pass
        # warnings.warn(f"Failed to infer shape for {dist_class.__name__}, "
        #               "falling back to expensive instance construction")

    # Rely on the underlying distribution's logic to infer the event_shape
    # given param domains.
    args = {k: dummy_numeric_array(domain)
            for k, domain in domains.items() if k != "value"}
    instance = dist_class(**args, validate_args=False)
    return instance.batch_shape, instance.event_shape


################################################################################
# Distribution Wrappers
################################################################################

def make_dist(backend_dist_class, param_names=(), generate_eager=True, generate_to_funsor=True):
    if not param_names:
        param_names = tuple(name for name in inspect.getfullargspec(backend_dist_class.__init__)[0][1:]
                            if name in backend_dist_class.arg_constraints)

    @makefun.with_signature("__init__(self, {}, value='value')".format(', '.join(param_names)))
    def dist_init(self, **kwargs):
        return Distribution.__init__(self, *tuple(kwargs[k] for k in self._ast_fields))

    dist_class = DistributionMeta(backend_dist_class.__name__.split("Wrapper_")[-1], (Distribution,), {
        'dist_class': backend_dist_class,
        '__init__': dist_init,
    })

    if generate_eager:
        eager.register(dist_class, *((Tensor,) * (len(param_names) + 1)))(dist_class.eager_log_prob)

    if generate_to_funsor:
        to_funsor.register(backend_dist_class)(functools.partial(backenddist_to_funsor, dist_class))

    return dist_class


FUNSOR_DIST_NAMES = [
    ('Beta', ('concentration1', 'concentration0')),
    ("Cauchy", ()),
    ("Chi2", ()),
    ('BernoulliProbs', ('probs',)),
    ('BernoulliLogits', ('logits',)),
    ('Binomial', ('total_count', 'probs')),
    ('Categorical', ('probs',)),
    ('CategoricalLogits', ('logits',)),
    ('Delta', ('v', 'log_density')),
    ('Dirichlet', ('concentration',)),
    ('DirichletMultinomial', ('concentration', 'total_count')),
    ("Exponential", ()),
    ('Gamma', ('concentration', 'rate')),
    ('GammaPoisson', ('concentration', 'rate')),
    ("Geometric", ("probs",)),
    ("Gumbel", ()),
    ("HalfCauchy", ()),
    ("HalfNormal", ()),
    ("Laplace", ()),
    ("LowRankMultivariateNormal", ()),
    ('Multinomial', ('total_count', 'probs')),
    ('MultivariateNormal', ('loc', 'scale_tril')),
    ('NonreparameterizedBeta', ('concentration1', 'concentration0')),
    ('NonreparameterizedDirichlet', ('concentration',)),
    ('NonreparameterizedGamma', ('concentration', 'rate')),
    ('NonreparameterizedNormal', ('loc', 'scale')),
    ('Normal', ('loc', 'scale')),
    ("Pareto", ()),
    ("Poisson", ()),
    ("StudentT", ()),
    ("Uniform", ()),
    ("VonMises", ()),
]


###############################################
# Converting backend Distributions to funsors
###############################################

def backenddist_to_funsor(funsor_dist_class, backend_dist, output=None, dim_to_name=None):
    params = [to_funsor(
            getattr(backend_dist, param_name),
            output=funsor_dist_class._infer_param_domain(
                param_name, getattr(getattr(backend_dist, param_name), "shape", ())),
            dim_to_name=dim_to_name)
        for param_name in funsor_dist_class._ast_fields if param_name != 'value']
    return funsor_dist_class(*params)


def indepdist_to_funsor(backend_dist, output=None, dim_to_name=None):
    dim_to_name = OrderedDict((dim - backend_dist.reinterpreted_batch_ndims, name)
                              for dim, name in dim_to_name.items())
    dim_to_name.update(OrderedDict((i, "_pyro_event_dim_{}".format(i))
                                   for i in range(-backend_dist.reinterpreted_batch_ndims, 0)))
    result = to_funsor(backend_dist.base_dist, dim_to_name=dim_to_name)
    for i in reversed(range(-backend_dist.reinterpreted_batch_ndims, 0)):
        name = "_pyro_event_dim_{}".format(i)
        result = funsor.terms.Independent(result, "value", name, "value")
    return result


def expandeddist_to_funsor(backend_dist, output=None, dim_to_name=None):

    funsor_base_dist = to_funsor(backend_dist.base_dist, output=output, dim_to_name=dim_to_name)
    if not dim_to_name:
        assert not backend_dist.batch_shape
        return funsor_base_dist

    name_to_dim = {name: dim for dim, name in dim_to_name.items()}
    raw_expanded_params = {}
    for name, funsor_param in funsor_base_dist.params.items():
        if name == "value":
            continue
        raw_param = to_data(funsor_param, name_to_dim=name_to_dim)
        raw_expanded_params[name] = ops.expand(raw_param, backend_dist.batch_shape + funsor_param.shape)

    raw_expanded_dist = type(backend_dist.base_dist)(**raw_expanded_params)
    return to_funsor(raw_expanded_dist, output, dim_to_name)


def maskeddist_to_funsor(backend_dist, output=None, dim_to_name=None):
    mask = to_funsor(ops.astype(backend_dist._mask, 'float32'), output=output, dim_to_name=dim_to_name)
    funsor_base_dist = to_funsor(backend_dist.base_dist, output=output, dim_to_name=dim_to_name)
    return mask * funsor_base_dist


# converts TransformedDistributions
def transformeddist_to_funsor(backend_dist, output=None, dim_to_name=None):
    dist_module = import_module(BACKEND_TO_DISTRIBUTIONS_BACKEND[get_backend()]).dist
    base_dist, transforms = backend_dist, []
    while isinstance(base_dist, dist_module.TransformedDistribution):
        transforms = base_dist.transforms + transforms
        base_dist = base_dist.base_dist
    funsor_base_dist = to_funsor(base_dist, output=output, dim_to_name=dim_to_name)
    # TODO make this work with transforms that change the output type
    transform = to_funsor(dist_module.transforms.ComposeTransform(transforms),
                          funsor_base_dist.inputs["value"], dim_to_name)
    _, inv_transform, ldj = funsor.delta.solve(transform, to_funsor("value", funsor_base_dist.inputs["value"]))
    return -ldj + funsor_base_dist(value=inv_transform)


class CoerceDistributionToFunsor:
    """
    Handler to reinterpret a backend distribution ``D`` as a corresponding
    funsor during ``type(D).__call__()`` in case any constructor args are
    funsors rather than backend tensors.

    Example usage::

        # in foo/distribution.py
        coerce_to_funsor = CoerceDistributionToFunsor("foo")

        class DistributionMeta(type):
            def __call__(cls, *args, **kwargs):
                result = coerce_to_funsor(cls, args, kwargs)
                if result is not None:
                    return result
                return super().__call__(*args, **kwargs)

        class Distribution(metaclass=DistributionMeta):
            ...

    :param str backend: Name of a funsor backend.
    """
    def __init__(self, backend):
        self.backend = backend

    @lazy_property
    def module(self):
        funsor.set_backend(self.backend)
        module_name = BACKEND_TO_DISTRIBUTIONS_BACKEND[self.backend]
        return importlib.import_module(module_name)

    def __call__(self, cls, args, kwargs):
        # Check whether distribution class takes any tensor inputs.
        arg_constraints = getattr(cls, "arg_constraints", None)
        if not arg_constraints:
            return

        # Check whether any tensor inputs are actually funsors.
        try:
            ast_fields = cls._funsor_ast_fields
        except AttributeError:
            ast_fields = cls._funsor_ast_fields = getargspec(cls.__init__)[0][1:]
        kwargs = {name: value for pairs in (zip(ast_fields, args), kwargs.items())
                  for name, value in pairs}
        if not any(isinstance(value, (str, Funsor))
                   for name, value in kwargs.items()
                   if name in arg_constraints):
            return

        # Check for a corresponding funsor class.
        try:
            funsor_cls = cls._funsor_cls
        except AttributeError:
            funsor_cls = getattr(self.module, cls.__name__, None)
            # resolve the issues Binomial/Multinomial are functions in NumPyro, which
            # fallback to either BinomialProbs or BinomialLogits
            if funsor_cls is None and cls.__name__.endswith("Probs"):
                funsor_cls = getattr(self.module, cls.__name__[:-5], None)
            cls._funsor_cls = funsor_cls
        if funsor_cls is None:
            warnings.warn("missing funsor for {}".format(cls.__name__),
                          RuntimeWarning)
            return

        # Coerce to funsor.
        return funsor_cls(**kwargs)


###############################################################
# Converting distribution funsors to backend distributions
###############################################################

@to_data.register(Distribution)
def distribution_to_data(funsor_dist, name_to_dim=None):
    funsor_event_shape = funsor_dist.value.output.shape

    # attempt to generically infer the independent output dimensions
    domains = {k: v.output for k, v in funsor_dist.params.items()}
    indep_shape, _ = infer_shapes(funsor_dist.dist_class, domains)

    params = []
    for param_name, funsor_param in zip(funsor_dist._ast_fields, funsor_dist._ast_values[:-1]):
        param = to_data(funsor_param, name_to_dim=name_to_dim)

        # infer the independent dimensions of each parameter separately, since we chose to keep them unbroadcasted
        param_event_shape = getattr(funsor_dist._infer_param_domain(param_name, funsor_param.output.shape), "shape", ())
        param_indep_shape = funsor_param.output.shape[:len(funsor_param.output.shape) - len(param_event_shape)]
        for i in range(max(0, len(indep_shape) - len(param_indep_shape))):
            # add singleton event dimensions, leave broadcasting/expanding to backend
            param = ops.unsqueeze(param, -1 - len(funsor_param.output.shape))

        params.append(param)

    pyro_dist = funsor_dist.dist_class(**dict(zip(funsor_dist._ast_fields[:-1], params)))
    pyro_dist = pyro_dist.to_event(max(len(funsor_event_shape) - len(pyro_dist.event_shape), 0))

    # TODO get this working for all backends
    if not isinstance(funsor_dist.value, Variable):
        if get_backend() != "torch":
            raise NotImplementedError("transformed distributions not yet supported under this backend,"
                                      "try set_backend('torch')")
        inv_value = funsor.delta.solve(funsor_dist.value, Variable("value", funsor_dist.value.output))[1]
        transforms = to_data(inv_value, name_to_dim=name_to_dim)
        backend_dist = import_module(BACKEND_TO_DISTRIBUTIONS_BACKEND[get_backend()]).dist
        pyro_dist = backend_dist.TransformedDistribution(pyro_dist, transforms)

    if pyro_dist.event_shape != funsor_event_shape:
        raise ValueError("Event shapes don't match, something went wrong")
    return pyro_dist


@to_data.register(Independent[typing.Union[Independent, Distribution], str, str, str])
def indep_to_data(funsor_dist, name_to_dim=None):
    if not isinstance(funsor_dist.fn, (Independent, Distribution, Gaussian)):
        raise NotImplementedError(f"cannot convert {funsor_dist} to data")
    name_to_dim = OrderedDict((name, dim - 1) for name, dim in name_to_dim.items())
    name_to_dim.update({funsor_dist.bint_var: -1})
    backend_dist = import_module(BACKEND_TO_DISTRIBUTIONS_BACKEND[get_backend()]).dist
    result = to_data(funsor_dist.fn, name_to_dim=name_to_dim)

    # collapse nested Independents into a single Independent for conversion
    reinterpreted_batch_ndims = 1
    while isinstance(result, backend_dist.Independent):
        result = result.base_dist
        reinterpreted_batch_ndims += 1

    return backend_dist.Independent(result, reinterpreted_batch_ndims)


@to_data.register(Gaussian)
def gaussian_to_data(funsor_dist, name_to_dim=None, normalized=False):
    if normalized:
        return to_data(funsor_dist.log_normalizer + funsor_dist, name_to_dim=name_to_dim)
    loc = ops.cholesky_solve(ops.unsqueeze(funsor_dist.info_vec, -1),
                             ops.cholesky(funsor_dist.precision)).squeeze(-1)
    int_inputs = OrderedDict((k, d) for k, d in funsor_dist.inputs.items() if d.dtype != "real")
    loc = to_data(Tensor(loc, int_inputs), name_to_dim)
    precision = to_data(Tensor(funsor_dist.precision, int_inputs), name_to_dim)
    backend_dist = import_module(BACKEND_TO_DISTRIBUTIONS_BACKEND[get_backend()])
    return backend_dist.MultivariateNormal.dist_class(loc, precision_matrix=precision)


@to_data.register(GaussianMixture)
def gaussianmixture_to_data(funsor_dist, name_to_dim=None):
    discrete, gaussian = funsor_dist.terms
    backend_dist = import_module(BACKEND_TO_DISTRIBUTIONS_BACKEND[get_backend()])
    cat = backend_dist.CategoricalLogits.dist_class(logits=to_data(
        discrete + gaussian.log_normalizer, name_to_dim=name_to_dim))
    mvn = to_data(gaussian, name_to_dim=name_to_dim)
    return cat, mvn


################################################
# Backend-agnostic distribution patterns
################################################

def Bernoulli(probs=None, logits=None, value='value'):
    """
    Wraps backend `Bernoulli` distributions.

    This dispatches to either `BernoulliProbs` or `BernoulliLogits`
    to accept either ``probs`` or ``logits`` args.

    :param Funsor probs: Probability of 1.
    :param Funsor value: Optional observation in ``{0,1}``.
    """
    backend_dist = import_module(BACKEND_TO_DISTRIBUTIONS_BACKEND[get_backend()])
    if probs is not None:
        probs = to_funsor(probs, output=Real)
        return backend_dist.BernoulliProbs(probs, value)  # noqa: F821
    if logits is not None:
        logits = to_funsor(logits, output=Real)
        return backend_dist.BernoulliLogits(logits, value)  # noqa: F821
    raise ValueError('Either probs or logits must be specified')


def LogNormal(loc, scale, value='value'):
    """
    Wraps backend `LogNormal` distributions.

    :param Funsor loc: Mean of the untransformed Normal distribution.
    :param Funsor scale: Standard deviation of the untransformed Normal
        distribution.
    :param Funsor value: Optional real observation.
    """
    loc, scale = to_funsor(loc), to_funsor(scale)
    y = to_funsor(value, output=loc.output)
    t = ops.exp
    x = t.inv(y)
    log_abs_det_jacobian = t.log_abs_det_jacobian(x, y)
    backend_dist = import_module(BACKEND_TO_DISTRIBUTIONS_BACKEND[get_backend()])
    return backend_dist.Normal(loc, scale, x) - log_abs_det_jacobian  # noqa: F821


def eager_beta(concentration1, concentration0, value):
    concentration = stack((concentration0, concentration1))
    value = stack((1 - value, value))
    backend_dist = import_module(BACKEND_TO_DISTRIBUTIONS_BACKEND[get_backend()])
    return backend_dist.Dirichlet(concentration, value=value)  # noqa: F821


def eager_binomial(total_count, probs, value):
    probs = stack((1 - probs, probs))
    value = stack((total_count - value, value))
    backend_dist = import_module(BACKEND_TO_DISTRIBUTIONS_BACKEND[get_backend()])
    return backend_dist.Multinomial(total_count, probs, value=value)  # noqa: F821


def eager_multinomial(total_count, probs, value):
    # Multinomial.log_prob() supports inhomogeneous total_count only by
    # avoiding passing total_count to the constructor.
    inputs, (total_count, probs, value) = align_tensors(total_count, probs, value)
    shape = broadcast_shape(total_count.shape + (1,), probs.shape, value.shape)
    probs = Tensor(ops.expand(probs, shape), inputs)
    value = Tensor(ops.expand(value, shape), inputs)
    if get_backend() == "torch":
        total_count = Number(ops.amax(total_count, None).item())  # Used by distributions validation code.
    else:
        total_count = Tensor(ops.expand(total_count, shape[:-1]), inputs)
    backend_dist = import_module(BACKEND_TO_DISTRIBUTIONS_BACKEND[get_backend()])
    return backend_dist.Multinomial.eager_log_prob(total_count, probs, value)  # noqa: F821


def eager_categorical_funsor(probs, value):
    return probs[value].log()


def eager_categorical_tensor(probs, value):
    value = probs.materialize(value)
    backend_dist = import_module(BACKEND_TO_DISTRIBUTIONS_BACKEND[get_backend()])
    return backend_dist.Categorical(probs=probs, value=value)  # noqa: F821


def eager_delta_tensor(v, log_density, value):
    # This handles event_dim specially, and hence cannot use the
    # generic Delta.eager_log_prob() method.
    assert v.output == value.output
    event_dim = len(v.output.shape)
    inputs, (v, log_density, value) = align_tensors(v, log_density, value)
    backend_dist = import_module(BACKEND_TO_DISTRIBUTIONS_BACKEND[get_backend()])
    data = backend_dist.Delta.dist_class(v, log_density, event_dim).log_prob(value)  # noqa: F821
    return Tensor(data, inputs)


def eager_delta_funsor_variable(v, log_density, value):
    assert v.output == value.output
    return funsor.delta.Delta(value.name, v, log_density)


def eager_delta_funsor_funsor(v, log_density, value):
    assert v.output == value.output
    return funsor.delta.Delta(v.name, value, log_density)


def eager_delta_variable_variable(v, log_density, value):
    return None


def eager_normal(loc, scale, value):
    assert loc.output == Real
    assert scale.output == Real
    assert value.output == Real
    if not is_affine(loc) or not is_affine(value):
        return None  # lazy

    info_vec = ops.new_zeros(scale.data, scale.data.shape + (1,))
    precision = ops.pow(scale.data, -2).reshape(scale.data.shape + (1, 1))
    log_prob = -0.5 * math.log(2 * math.pi) - ops.log(scale).sum()
    inputs = scale.inputs.copy()
    var = gensym('value')
    inputs[var] = Real
    gaussian = log_prob + Gaussian(info_vec, precision, inputs)
    return gaussian(**{var: value - loc})


def eager_mvn(loc, scale_tril, value):
    assert len(loc.shape) == 1
    assert len(scale_tril.shape) == 2
    assert value.output == loc.output
    if not is_affine(loc) or not is_affine(value):
        return None  # lazy

    info_vec = ops.new_zeros(scale_tril.data, scale_tril.data.shape[:-1])
    precision = ops.cholesky_inverse(scale_tril.data)
    scale_diag = Tensor(ops.diagonal(scale_tril.data, -1, -2), scale_tril.inputs)
    log_prob = -0.5 * scale_diag.shape[0] * math.log(2 * math.pi) - ops.log(scale_diag).sum()
    inputs = scale_tril.inputs.copy()
    var = gensym('value')
    inputs[var] = Reals[scale_diag.shape[0]]
    gaussian = log_prob + Gaussian(info_vec, precision, inputs)
    return gaussian(**{var: value - loc})


def eager_beta_bernoulli(red_op, bin_op, reduced_vars, x, y):
    backend_dist = import_module(BACKEND_TO_DISTRIBUTIONS_BACKEND[get_backend()])
    return eager_dirichlet_multinomial(red_op, bin_op, reduced_vars, x,
                                       backend_dist.Binomial(total_count=1, probs=y.probs, value=y.value))


def eager_dirichlet_categorical(red_op, bin_op, reduced_vars, x, y):
    dirichlet_reduction = x.input_vars & reduced_vars
    if dirichlet_reduction:
        backend_dist = import_module(BACKEND_TO_DISTRIBUTIONS_BACKEND[get_backend()])
        identity = Tensor(ops.new_eye(funsor.tensor.get_default_prototype(), x.concentration.shape))
        return backend_dist.DirichletMultinomial(concentration=x.concentration,
                                                 total_count=1,
                                                 value=identity[y.value])
    else:
        return eager(Contraction, red_op, bin_op, reduced_vars, (x, y))


def eager_dirichlet_multinomial(red_op, bin_op, reduced_vars, x, y):
    dirichlet_reduction = x.input_vars & reduced_vars
    if dirichlet_reduction:
        backend_dist = import_module(BACKEND_TO_DISTRIBUTIONS_BACKEND[get_backend()])
        return backend_dist.DirichletMultinomial(concentration=x.concentration,
                                                 total_count=y.total_count,
                                                 value=y.value)
    else:
        return eager(Contraction, red_op, bin_op, reduced_vars, (x, y))


def eager_plate_multinomial(op, x, reduced_vars):
    if not reduced_vars.isdisjoint(x.probs.input_vars):
        return None
    if not reduced_vars.issubset(x.value.input_vars):
        return None

    backend_dist = import_module(BACKEND_TO_DISTRIBUTIONS_BACKEND[get_backend()])
    total_count = x.total_count
    for v in reduced_vars:
        if v.name in total_count.inputs:
            total_count = total_count.reduce(ops.add, v)
        else:
            total_count = total_count * v.output.size
    return backend_dist.Multinomial(total_count=total_count,
                                    probs=x.probs,
                                    value=x.value.reduce(ops.add, reduced_vars))


def _log_beta(x, y):
    return ops.lgamma(x) + ops.lgamma(y) - ops.lgamma(x + y)


def eager_gamma_gamma(red_op, bin_op, reduced_vars, x, y):
    gamma_reduction = x.input_vars & reduced_vars
    if gamma_reduction:
        unnormalized = (y.concentration - 1) * ops.log(y.value) \
            - (y.concentration + x.concentration) * ops.log(y.value + x.rate)
        const = -x.concentration * ops.log(x.rate) + _log_beta(y.concentration, x.concentration)
        return unnormalized - const
    else:
        return eager(Contraction, red_op, bin_op, reduced_vars, (x, y))


def eager_gamma_poisson(red_op, bin_op, reduced_vars, x, y):
    gamma_reduction = x.input_vars & reduced_vars
    if gamma_reduction:
        backend_dist = import_module(BACKEND_TO_DISTRIBUTIONS_BACKEND[get_backend()])
        return backend_dist.GammaPoisson(concentration=x.concentration,
                                         rate=x.rate,
                                         value=y.value)
    else:
        return eager(Contraction, red_op, bin_op, reduced_vars, (x, y))


def eager_dirichlet_posterior(op, c, z):
    if (z.concentration is c.terms[0].concentration) and (c.terms[1].total_count is z.total_count):
        backend_dist = import_module(BACKEND_TO_DISTRIBUTIONS_BACKEND[get_backend()])
        return backend_dist.Dirichlet(
            concentration=z.concentration + c.terms[1].value,
            value=c.terms[0].value)
    else:
        return None
