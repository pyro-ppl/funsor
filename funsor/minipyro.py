"""
Mini Pyro
---------

This file contains a minimal implementation of the Pyro Probabilistic
Programming Language. The API (method signatures, etc.) match that of
the full implementation as closely as possible.

An accompanying example that makes use of this implementation can be
found at examples/minipyro.py.
"""
from __future__ import absolute_import, division, print_function

from collections import OrderedDict, defaultdict

import torch
from multipledispatch import dispatch

import funsor
import funsor.ops as ops
from funsor.domains import bint
from funsor.terms import Funsor, Number, Variable

from .handlers import HANDLER_STACK, Handler, Message, apply_stack, effectful


class Sample(Message):
    pass


class Param(Message):
    pass


class Barrier(Message):
    pass


class Ground(Message):
    pass


PARAM_STORE = {}


def get_param_store():
    return PARAM_STORE


class trace(Handler):
    def __enter__(self):
        super(trace, self).__enter__()
        self.trace = OrderedDict()
        return self.trace

    # trace illustrates why we need postprocess in addition to process:
    # We only want to record a value after all other effects have been applied
    @dispatch(Sample)
    def postprocess(self, msg):
        assert msg["name"] is not None and \
            msg["name"] not in self.trace, \
            "all sites must have unique names"
        self.trace[msg["name"]] = msg.copy()
        return msg

    def get_trace(self, *args, **kwargs):
        self(*args, **kwargs)
        return self.trace


# A second example of an effect handler for setting the value at a sample site.
# This illustrates why effect handlers are a useful PPL implementation technique:
# We can compose trace and replay to replace values but preserve distributions,
# allowing us to compute the joint probability density of samples under a model.
# See the definition of elbo(...) below for an example of this pattern.
class replay(Handler):
    def __init__(self, fn=None, guide_trace=None):
        self.guide_trace = guide_trace
        super(replay, self).__init__(fn)

    @dispatch(object)
    def process(self, msg):
        return super(replay, self).process(msg)

    @dispatch(Sample)
    def process(self, msg):
        if msg["name"] in self.guide_trace:
            msg["value"] = self.guide_trace[msg["name"]]["value"]
        return msg


class block(Handler):
    """
    This allows the selective application of effect handlers to different parts
    of a model.  Sites hidden by block will only have the handlers below block
    on the ``HANDLER_STACK`` applied, allowing inference or other effectful
    computations to be nested inside models.
    """
    def __init__(self, fn=None, hide_fn=lambda msg: True):
        self.hide_fn = hide_fn
        super(block, self).__init__(fn)

    @dispatch(Message)
    def process(self, msg):
        if self.hide_fn(msg):
            msg["stop"] = True
        return msg


class plate(Handler):
    def __init__(self, fn, name, size):
        assert isinstance(name, str)
        assert isinstance(size, int) and size > 0
        self.name = name
        self.size = size
        super(plate, self).__init__(fn)

    @dispatch(object)
    def process(self, msg):
        return super(plate, self).process(msg)

    @dispatch(Sample)
    def process(self, msg):
        msg.setdefault("plates", set()).add(self.name)
        msg["fn"] = msg["fn"].expand(OrderedDict([(self.name, bint(self.size))]))
        return msg

    def __iter__(self):
        return range(self.size)


def sample(fn, obs=None, name=None):
    """
    This is an effectful version of ``Distribution.sample(...)``.  When any
    effect handlers are active, it constructs an initial message and calls
    ``apply_stack``.
    """
    assert isinstance(fn, Funsor)

    # if there are no active Handlers, we just create a lazy compute graph.
    if not HANDLER_STACK:
        return Variable(name, fn.output)

    # Otherwise, we initialize a message...
    initial_msg = Sample(**{
        "name": name,
        "fn": fn,
        "args": (),
        "kwargs": {},
        "value": obs,
    })

    # ...and use apply_stack to send it to the Handlers
    msg = apply_stack(initial_msg)
    return msg["value"]


def param(init_value=None, name=None):
    """
    This is an effectful version of ``PARAM_STORE.setdefault``. When any effect
    handlers are active, it constructs an initial message and calls
    ``apply_stack``.
    """

    if init_value is None and name is None:
        raise ValueError("empty args to param")

    def fn(init_value):
        value = PARAM_STORE.setdefault(name, init_value)
        value.requires_grad_()
        return value

    # if there are no active Handlers, we just draw a sample and return it as expected:
    if not HANDLER_STACK:
        return fn(init_value)

    # Otherwise, we initialize a message...
    initial_msg = Param(**{
        "fn": fn,
        "args": (init_value,),
        "value": None,
    })

    # ...and use apply_stack to send it to the Handlers
    msg = apply_stack(initial_msg)
    return msg["value"]


class SelectiveHandler(Handler):
    def __init__(self, fn=None, match_fn=None):
        self.match_fn = (lambda msg: True) if match_fn is None else match_fn
        super(SelectiveHandler, self).__init__(fn=fn)


class deferred(SelectiveHandler):

    @dispatch(object)
    def process(self, msg):
        return super(deferred, self).process(msg)

    @dispatch(Sample)
    def process(self, msg):
        if msg["value"] is not None and self.match_fn(msg):
            msg["value"] = Variable(msg["name"], msg["fn"].output)
        return msg


class monte_carlo(SelectiveHandler):

    @dispatch(object)
    def process(self, msg):
        return super(monte_carlo, self).process(msg)

    @dispatch(Sample)
    def process(self, msg):
        if msg["value"] is not None and self.match_fn(msg):
            msg["value"] = msg["fn"].sample()
        return msg


class log_joint(Handler):
    """
    Tracks log joint density during delayed sampling.
    """

    def __enter__(self):
        self.log_probs = defaultdict(lambda: Number(0))  # plate set -> log_prob
        self.samples = OrderedDict()  # site name -> value
        return self

    @property
    def log_prob(self):
        if any(self.log_probs.keys()):
            # Assume the program is finished and
            # completely contract out all variables.
            factors = list(self.log_probs.values())
            plates = frozenset().union(*self.log_probs)
            samples = frozenset().union(*(f.inputs for f in factors)) - plates
            log_prob = funsor.contract.sum_product(
                sum_op=ops.logaddexp,
                prod_op=ops.add,
                factors=factors,
                sum_vars=samples,
                prod_vars=plates)
            self.log_probs[frozenset()] = log_prob
        return self.log_probs[frozenset()]

    @dispatch(object)
    def process(self, msg):
        return super(log_joint, self).process(msg)

    @dispatch(Sample)
    def process(self, msg):
        assert msg["value"] is not None
        self.samples[msg["name"]] = msg["value"]
        plates = frozenset(msg.get("plates", ()))
        self.log_probs[plates] += msg["fn"](msg["value"])
        return msg

    @dispatch(Barrier)
    def process(self, msg):
        # Collect complete set of free variables, both delayed samples and plates.
        free_vars = frozenset(v for f in self.log_probs.values() for v in f.inputs)
        plates = frozenset().union(*self.log_probs)
        assert all(v in self.samples or v in plates for v in free_vars)

        # Collect the subset of variables that are exposed downstream.
        live_funsors = []
        _recursive_map(live_funsors.append, msg["value"])
        live_vars = frozenset(v for f in live_funsors for v in f.inputs)
        assert live_vars.issubset(free_vars)

        # Compute variables that are safe to eliminate.
        dead_vars = free_vars - live_vars
        dead_samples = dead_vars.intersection(self.samples)
        dead_plates = dead_vars.intersection(plates)

        # Eliminate samples and plates via tensor variable elimination.
        # This is analagous to pyro 0.3's contract_tensor_tree().
        with funsor.trace_monte_carlo_approximations() as subs:
            factors = funsor.contract.partial_sum_product(
                sum_op=ops.logaddexp,
                prod_op=ops.add,
                factors=self.log_probs.values(),
                sum_vars=dead_samples,
                prod_vars=dead_plates)
        self.log_probs.clear()
        for f in factors:
            self.log_probs[plates.intersection(f.inputs)] += f

        # The logaddexp reductions during elimination may have been interepreted as
        # Monte Carlo sampling, in which case we need to update self.samples and msg["value"].
        if subs:
            for name, sample in self.samples.items():
                self.samples[name] = sample(**subs)
            msg["value"] = _recursive_map(lambda x: x(**subs), msg["value"])

        return msg

    @dispatch(Ground)
    def process(self, msg):
        # Determine whether any delayed samples need to be marginalized.
        value = msg["value"]
        plates = frozenset().union(*self.log_probs)
        delayed_samples = frozenset(value.inputs) - plates
        if delayed_samples:
            with funsor.trace_monte_carlo_approximations() as subs:
                self.log_probs = funsor.einsum.partial_tensor_variable_elimination(
                    sum_op=ops.logaddexp,
                    prod_op=ops.add,
                    sum_vars=delayed_samples,
                    prod_vars=frozenset(),
                    funsor_tree=self.log_probs)
            self.samples.update(subs)
            msg["value"] = value(**subs)
            context = msg["context"]
            for key, value in list(context.items()):
                if isinstance(value, Funsor):
                    context[key] = value(**subs)
        return msg


@effectful(Barrier)
def barrier(state):
    """
    Declaration that behavior after this point in a program depends on behavior
    before this point in a program only through the passed ``state`` object,
    which can be a :class:`~funsor.Funsor` or recursive structure built from
    funsors via ``tuple`` or non-funsor keyed ``dict``.

    Example::

       x = 0
       for t in range(100):
           x = pyro.sample("x_{}".format(t), trans(x))
           x = pyro.barrier(x)  # it is now safe to marginalize past xs
           pyro.sample("y_{}".format(t), emit(x), obs=data[t])
    """
    # if there are no active Handlers, we just return the state
    return state


def _recursive_map(fn, x):
    if isinstance(x, funsor.Funsor):
        return fn(x)
    if isinstance(x, tuple):
        return tuple(fn(y) for y in x)
    if isinstance(x, dict):
        return {k: fn(v) for k, v in x.items()}


@effectful(Ground)
def ground(value, context):
    """
    Sample enough deferred random variables so that ``value`` is ground,
    and update ``context`` with the new samples. Typically ``context`` is
    ``locals()`` as called in a small model, or a global dict storing all
    random state in a larger model. This is typically used in a
    :class:`deferred` context.

    This is like ``value()`` in the Birch probabilistic programming language.

    Example::

        with pyro.deferred():
            # ...draw deferred samples...
            x = pyro.ground(x, locals())
            if x > 0:  # requires x to be a ground value
                # ...do stuff...

    :param Funsor value: A funsor possibly depending on delayed sample sites.
    :param dict context: A dict containing all other random state.
    :return: A version of ``value`` with all deferred variables sampled.
    :rtype: Funsor
    """
    assert isinstance(value, Funsor)
    assert isinstance(context, dict)
    return value


def elbo(model, guide, *args, **kwargs):
    """
    This is an attempt to compute a deferred elbo.
    """
    # sample guide
    with log_joint() as guide_joint:
        guide(*args, **kwargs)
        # FIXME This is only correct for reparametrized sites.
        # FIXME do not marginalize; instead sample.
        q = guide_joint.log_prob.reduce(ops.logaddexp)
    tr = guide_joint.samples
    tr.update(funsor.backward(ops.sample, q))  # force deferred samples?

    # replay model against guide
    with log_joint() as model_joint, replay(guide_trace=tr):
        model(*args, **kwargs)
        p = funsor.eval(model_joint.log_prob.reduce(ops.logaddexp))

    elbo = p - q
    return -elbo  # negate, for use as loss


class SVI(object):
    """
    This is a unified interface for stochastic variational inference in Pyro.
    The actual construction of the loss is taken care of by `loss`.
    See http://docs.pyro.ai/en/stable/inference_algos.html
    """
    def __init__(self, model, guide, optim, loss):
        self.model = model
        self.guide = guide
        self.optim = optim
        self.loss = loss

    # This method handles running the model and guide, constructing the loss
    # function, and taking a gradient step.
    def step(self, *args, **kwargs):
        # This wraps both the call to `model` and `guide` in a `trace` so that
        # we can record all the parameters that are encountered. Note that
        # further tracing occurs inside of `loss`.
        with trace() as param_capture:
            # We use block here to allow tracing to record parameters only.
            with block(hide_fn=lambda msg: msg["type"] == "sample"):
                loss = self.loss(self.model, self.guide, *args, **kwargs)
        # Differentiate the loss.
        loss.backward()
        # Grab all the parameters from the trace.
        params = [site["value"] for site in param_capture.values()]
        # Take a step w.r.t. each parameter in params.
        self.optim(params)
        # Zero out the gradients so that they don't accumulate.
        for p in params:
            p.grad = p.new_zeros(p.shape)
        return loss.item()


class Adam(object):
    """
    This is a thin wrapper around the `torch.optim.Adam` class that
    dynamically generates optimizers for dynamically generated parameters.
    See http://docs.pyro.ai/en/stable/optimization.html
    """
    def __init__(self, optim_args):
        self.optim_args = optim_args
        # Each parameter will get its own optimizer, which we keep track
        # of using this dictionary keyed on parameters.
        self.optim_objs = {}

    def __call__(self, params):
        for param in params:
            # If we've seen this parameter before, use the previously
            # constructed optimizer.
            if param in self.optim_objs:
                optim = self.optim_objs[param]
            # If we've never seen this parameter before, construct
            # an Adam optimizer and keep track of it.
            else:
                optim = torch.optim.Adam([param.data], **self.optim_args)
                self.optim_objs[param] = optim
            # Take a gradient step for the parameter param.
            optim.step()


__all__ = [
    'Adam',
    'barrier',
    'block',
    'deferred',
    'elbo',
    'get_param_store',
    'ground',
    'param',
    'plate',
    'replay',
    'sample',
    'trace',
]
