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

from collections import OrderedDict

import funsor
import funsor.ops as ops
from funsor.terms import Funsor, Variable
from .handlers import apply_stack, Handler, HANDLER_STACK, Label


class Sample(Label):
    pass


class Param(Label):
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
    def postprocess(self, msg):
        if isinstance(msg["label"], Sample):
            assert msg["label"].name is not None and \
                msg["label"].name not in self.trace, \
                "all sites must have unique names"
            self.trace[msg["label"].name] = msg.copy()

    def get_trace(self, *args, **kwargs):
        self(*args, **kwargs)
        return self.trace


# A second example of an effect handler for setting the value at a sample site.
# This illustrates why effect handlers are a useful PPL implementation technique:
# We can compose trace and replay to replace values but preserve distributions,
# allowing us to compute the joint probability density of samples under a model.
# See the definition of elbo(...) below for an example of this pattern.
class replay(Handler):
    def __init__(self, fn, guide_trace):
        self.guide_trace = guide_trace
        super(replay, self).__init__(fn)

    def process(self, msg):
        if isinstance(msg["label"], Sample) and msg["label"].name in self.guide_trace:
            msg["value"] = self.guide_trace[msg["label"].name]["value"]


# block allows the selective application of effect handlers to different parts of a model.
# Sites hidden by block will only have the handlers below block on the HANDLER_STACK applied,
# allowing inference or other effectful computations to be nested inside models.
class block(Handler):
    def __init__(self, fn=None, hide_fn=lambda msg: True):
        self.hide_fn = hide_fn
        super(block, self).__init__(fn)

    def process(self, msg):
        if self.hide_fn(msg):
            msg["stop"] = True


# This limited implementation of PlateHandler only implements broadcasting.
class plate(Handler):
    def __init__(self, fn, size, dim, name):
        assert dim < 0
        self.size = size
        self.dim = dim
        super(plate, self).__init__(fn)

    def process(self, msg):
        if isinstance(msg["label"], Sample):
            batch_shape = msg["fn"].batch_shape
            if len(batch_shape) < -self.dim or batch_shape[self.dim] != self.size:
                batch_shape = [1] * (-self.dim - len(batch_shape)) + list(batch_shape)
                batch_shape[self.dim] = self.size
                msg["fn"] = msg["fn"].expand(tuple(batch_shape))

    def __iter__(self):
        return range(self.size)


# sample is an effectful version of Distribution.sample(...)
# When any effect handlers are active, it constructs an initial message and calls apply_stack.
def sample(fn, obs=None, name=None):

    # if there are no active Handlers, we just draw a sample and return it as expected:
    if not HANDLER_STACK:
        return fn.sample('value')

    # Otherwise, we initialize a message...
    initial_msg = {
        "type": Sample(name=name),
        "fn": fn,
        "args": (),
        "kwargs": {},
        "value": obs,
    }

    # ...and use apply_stack to send it to the Handlers
    msg = apply_stack(initial_msg)
    return msg["value"]


# param is an effectful version of PARAM_STORE.setdefault
# When any effect handlers are active, it constructs an initial message and calls apply_stack.
def param(init_value=None, name=None):

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
    initial_msg = {
        "type": Param(name=name),
        "fn": fn,
        "args": (init_value,),
        "value": None,
    }

    # ...and use apply_stack to send it to the Handlers
    msg = apply_stack(initial_msg)
    return msg["value"]


class deferred(Handler):

    def process(self, msg):
        if isinstance(msg["label"], Sample) and msg["value"] is not None:
            msg["value"] = Variable(msg["label"].name, msg["fn"].schema["value"])
        return msg


class log_joint(Handler):
    """
    Tracks log joint density during delayed sampling.
    """

    def __enter__(self):
        self.log_prob = funsor.to_funsor(0.)

    def process(self, msg):
        if isinstance(msg["label"], Sample):
            assert msg["value"] is not None
            self.log_prob += msg["fn"](msg["value"])

        elif isinstance(msg["label"], Ground):
            value = msg["value"]
            if not isinstance(value, (funsor.Number, funsor.Tensor)):
                log_prob = self.log_prob.reduce(ops.sample, value.dims)
                with funsor.adjoints():
                    self.log_prob = funsor.eval(log_prob)
                subs = funsor.backward(ops.sample, self.log_prob, value.dims)
                msg["value"] = value(**subs)
                context = msg["context"]
                for key, value in list(context.items()):
                    if isinstance(value, Funsor):
                        context[key] = value(**subs)

        return msg


class Ground(Label):
    pass


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

    # if there are no active Messengers, we just return the value
    if not HANDLER_STACK:
        return value

    # Otherwise, we initialize a message...
    initial_msg = {
        "type": Ground(),
        "context": context,
        "value": value,
    }

    # ...and use apply_stack to send it to the Messengers
    msg = apply_stack(initial_msg)
    return msg["value"]
