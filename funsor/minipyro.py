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

from .terms import Variable

# Pyro keeps track of two kinds of global state:
# i)  The effect handler stack, which enables non-standard interpretations of
#     Pyro primitives like sample();
#     See http://docs.pyro.ai/en/0.3.0-release/poutine.html
# ii) Trainable parameters in the Pyro ParamStore;
#     See http://docs.pyro.ai/en/0.3.0-release/parameters.html

PYRO_STACK = []
PARAM_STORE = {}


def get_param_store():
    return PARAM_STORE


# The base effect handler class (called Messenger here for consistency with Pyro).
class Messenger(object):
    def __init__(self, fn=None):
        self.fn = fn

    # Effect handlers push themselves onto the PYRO_STACK.
    # Handlers earlier in the PYRO_STACK are applied first.
    def __enter__(self):
        PYRO_STACK.append(self)

    def __exit__(self, *args, **kwargs):
        assert PYRO_STACK[-1] is self
        PYRO_STACK.pop()

    def process_message(self, msg):
        pass

    def postprocess_message(self, msg):
        pass

    def __call__(self, *args, **kwargs):
        with self:
            return self.fn(*args, **kwargs)


# A first useful example of an effect handler.
# trace records the inputs and outputs of any primitive site it encloses,
# and returns a dictionary containing that data to the user.
class trace(Messenger):
    def __enter__(self):
        super(trace, self).__enter__()
        self.trace = OrderedDict()
        return self.trace

    # trace illustrates why we need postprocess_message in addition to process_message:
    # We only want to record a value after all other effects have been applied
    def postprocess_message(self, msg):
        assert msg["name"] not in self.trace, "all sites must have unique names"
        self.trace[msg["name"]] = msg.copy()

    def get_trace(self, *args, **kwargs):
        self(*args, **kwargs)
        return self.trace


# A second example of an effect handler for setting the value at a sample site.
# This illustrates why effect handlers are a useful PPL implementation technique:
# We can compose trace and replay to replace values but preserve distributions,
# allowing us to compute the joint probability density of samples under a model.
# See the definition of elbo(...) below for an example of this pattern.
class replay(Messenger):
    def __init__(self, fn, guide_trace):
        self.guide_trace = guide_trace
        super(replay, self).__init__(fn)

    def process_message(self, msg):
        if msg["name"] in self.guide_trace:
            msg["value"] = self.guide_trace[msg["name"]]["value"]


# block allows the selective application of effect handlers to different parts of a model.
# Sites hidden by block will only have the handlers below block on the PYRO_STACK applied,
# allowing inference or other effectful computations to be nested inside models.
class block(Messenger):
    def __init__(self, fn=None, hide_fn=lambda msg: True):
        self.hide_fn = hide_fn
        super(block, self).__init__(fn)

    def process_message(self, msg):
        if self.hide_fn(msg):
            msg["stop"] = True


# This limited implementation of PlateMessenger only implements broadcasting.
class plate(Messenger):
    def __init__(self, fn, size, dim, name):
        assert dim < 0
        self.size = size
        self.dim = dim
        super(plate, self).__init__(fn)

    def process_message(self, msg):
        if msg["type"] == "sample":
            batch_shape = msg["fn"].batch_shape
            if len(batch_shape) < -self.dim or batch_shape[self.dim] != self.size:
                batch_shape = [1] * (-self.dim - len(batch_shape)) + list(batch_shape)
                batch_shape[self.dim] = self.size
                msg["fn"] = msg["fn"].expand(tuple(batch_shape))

    def __iter__(self):
        return range(self.size)


# apply_stack is called by pyro.sample and pyro.param.
# It is responsible for applying each Messenger to each effectful operation.
def apply_stack(msg):
    for pointer, handler in enumerate(reversed(PYRO_STACK)):
        handler.process_message(msg)
        # When a Messenger sets the "stop" field of a message,
        # it prevents any Messengers above it on the stack from being applied.
        if msg.get("stop"):
            break
    if msg["value"] is None:
        msg["value"] = msg["fn"](*msg["args"])

    # A Messenger that sets msg["stop"] == True also prevents application
    # of postprocess_message by Messengers above it on the stack
    # via the pointer variable from the process_message loop
    for handler in PYRO_STACK[-pointer-1:]:
        handler.postprocess_message(msg)
    return msg


# sample is an effectful version of Distribution.sample(...)
# When any effect handlers are active, it constructs an initial message and calls apply_stack.
def sample(fn, obs=None, name=None):

    # if there are no active Messengers, we just draw a sample and return it as expected:
    if not PYRO_STACK:
        args, log_prob = fn.sample(['value'])
        return args['value']

    # Otherwise, we initialize a message...
    initial_msg = {
        "type": "sample",
        "name": name if name is not None else "sample",
        "fn": fn,
        "args": (),
        "value": obs,
    }

    # ...and use apply_stack to send it to the Messengers
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

    # if there are no active Messengers, we just draw a sample and return it as expected:
    if not PYRO_STACK:
        return fn(init_value)

    # Otherwise, we initialize a message...
    initial_msg = {
        "type": "param",
        "name": name,
        "fn": fn,
        "args": (init_value,),
        "value": None,
    }

    # ...and use apply_stack to send it to the Messengers
    msg = apply_stack(initial_msg)
    return msg["value"]


class deferred(Messenger):

    def process_message(self, msg):
        if msg["type"] == "sample" and msg["value"] is not None:
            msg["value"] = Variable(msg["name"], msg["fn"].schema["value"])
        return msg
