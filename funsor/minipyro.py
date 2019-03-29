"""
Mini Pyro
---------

This file contains a minimal implementation of the Pyro Probabilistic
Programming Language. The API (method signatures, etc.) match that of
the full implementation as closely as possible. This file is independent
of the rest of Pyro, with the exception of the :mod:`pyro.distributions`
module.

An accompanying example that makes use of this implementation can be
found at examples/minipyro.py.
"""
from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import torch

import funsor

# Pyro keeps track of two kinds of global state:
# i)  The effect handler stack, which enables non-standard interpretations of
#     Pyro primitives like sample();
#     See http://docs.pyro.ai/en/0.3.1/poutine.html
# ii) Trainable parameters in the Pyro ParamStore;
#     See http://docs.pyro.ai/en/0.3.1/parameters.html

PYRO_STACK = []
PARAM_STORE = {}


def get_param_store():
    return PARAM_STORE


def clear_param_store():
    PARAM_STORE.clear()


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


class log_joint(Messenger):
    def __enter__(self):
        super(log_joint, self).__enter__()
        self.log_factors = OrderedDict()
        self.plates = set()
        return self

    def process_message(self, msg):
        if msg["type"] != "sample":
            return None
        if msg["value"] is None:
            msg["value"] = funsor.Variable(msg["name"], msg["fn"].inputs["value"])

    def postprocess_message(self, msg):
        if msg["type"] != "sample":
            return None
        assert msg["name"] not in self.log_factors, "all sites must have unique names"
        log_prob = msg["fn"](value=msg["value"])
        self.log_factors[msg["name"]] = log_prob
        self.plates.update(msg["cond_indep_stack"].values())  # maps dim to name

    def contract(self):
        return funsor.sum_product.sum_product(
            sum_op=funsor.ops.logaddexp,
            prod_op=funsor.ops.add,
            factors=list(self.log_factors.values()),
            plates=frozenset(self.plates),
            eliminate=frozenset(self.log_factors.keys()))


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
class PlateMessenger(Messenger):
    def __init__(self, fn, size, dim, name):
        assert dim < 0
        self.size = size
        self.dim = dim
        self.name = name
        super(PlateMessenger, self).__init__(fn)

    def process_message(self, msg):
        if msg["type"] == "sample":
            assert self.dim not in msg["cond_indep_stack"]
            msg["cond_indep_stack"][self.dim] = self.name

            if msg["value"] is not None:
                value = msg["value"]
                if not isinstance(value, funsor.Funsor):
                    assert isinstance(value, torch.Tensor)
                    output = msg["fn"].inputs["value"]
                    event_shape = output.shape
                    batch_shape = value.shape[:value.dim() - len(event_shape)]
                    inputs = OrderedDict()
                    data = value
                    for dim, size in enumerate(batch_shape):
                        if size == 1:
                            data = data.squeeze(dim - value.dim())
                        else:
                            name = msg["cond_indep_stack"][dim - len(batch_shape)]
                            inputs[name] = funsor.bint(size)
                    value = funsor.torch.Tensor(data, inputs, output.dtype)
                    assert value.output == output
                    msg["value"] = value

            # TODO expand function
            # batch_shape = msg["fn"].batch_shape
            # if len(batch_shape) < -self.dim or batch_shape[self.dim] != self.size:
            #     batch_shape = [1] * (-self.dim - len(batch_shape)) + list(batch_shape)
            #     batch_shape[self.dim] = self.size
            #     msg["fn"] = msg["fn"].expand(torch.Size(batch_shape))

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
def sample(name, fn, obs=None):

    # if there are no active Messengers, we just draw a sample and return it as expected:
    if not PYRO_STACK:
        raise NotImplementedError('Funsor cannot sample')
        # return fn()

    # Otherwise, we initialize a message...
    initial_msg = {
        "type": "sample",
        "name": name,
        "fn": fn,
        "args": (),
        "value": obs,
        "cond_indep_stack": {},  # maps dim to name
    }

    # ...and use apply_stack to send it to the Messengers
    msg = apply_stack(initial_msg)
    return msg["value"]


# param is an effectful version of PARAM_STORE.setdefault
# When any effect handlers are active, it constructs an initial message and calls apply_stack.
def param(name, init_value=None):

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


# boilerplate to match the syntax of actual pyro.plate:
def plate(name, size, dim):
    return PlateMessenger(fn=None, size=size, dim=dim, name=name)


# This is a thin wrapper around the `torch.optim.Adam` class that
# dynamically generates optimizers for dynamically generated parameters.
# See http://docs.pyro.ai/en/0.3.1/optimization.html
class Adam(object):
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
                optim = torch.optim.Adam([param], **self.optim_args)
                self.optim_objs[param] = optim
            # Take a gradient step for the parameter param.
            optim.step()


# This is a unified interface for stochastic variational inference in Pyro.
# The actual construction of the loss is taken care of by `loss`.
# See http://docs.pyro.ai/en/0.3.1/inference_algos.html
class SVI(object):
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
            with block(hide_fn=lambda msg: msg["type"] != "param"):
                with funsor.montecarlo.monte_carlo_interpretation():
                    loss = self.loss(self.model, self.guide, *args, **kwargs)
        # Differentiate the loss.
        loss.data.backward()
        # Grab all the parameters from the trace.
        params = [site["value"] for site in param_capture.values()]
        # Take a step w.r.t. each parameter in params.
        self.optim(params)
        # Zero out the gradients so that they don't accumulate.
        for p in params:
            p.grad = p.new_zeros(p.shape)
        return loss.item()


# TODO(eb8680) Replace this with funsor.Expectation.
def Expectation(log_probs, costs, sum_vars, prod_vars):
    probs = [p.exp() for p in log_probs]
    result = 0
    for cost in costs:
        result += funsor.sum_product.sum_product(
                sum_op=funsor.ops.add,
                prod_op=funsor.ops.mul,
                factors=probs + [cost],
                plates=prod_vars,
                eliminate=prod_vars | sum_vars)
    return result


# This is a basic implementation of the Evidence Lower Bound, which is the
# fundamental objective in Variational Inference.
# See http://pyro.ai/examples/svi_part_i.html for details.
# This implementation is closest to TraceEnum_ELBO insofar as it uses
# a Dice estimator with coarse plate-based dependency structure.
def elbo(model, guide, *args, **kwargs):
    with log_joint() as guide_log_joint:
        guide(*args, **kwargs)
    with log_joint() as model_log_joint:
        model(*args, **kwargs)
    plates = frozenset(guide_log_joint.plates | model_log_joint.plates)
    sum_vars = frozenset().union(guide_log_joint.log_factors,
                                 model_log_joint.log_factors)

    # Accumulate costs from model and guide and log_probs from guide.
    # Cf. pyro.infer.traceenum_elbo._compute_dice_elbo()
    # https://github.com/pyro-ppl/pyro/blob/0.3.0/pyro/infer/traceenum_elbo.py#L119
    costs = []
    log_probs = []
    for p in model_log_joint.log_factors.values():
        costs.append(p)
    for q in guide_log_joint.log_factors.values():
        costs.append(-q)
        log_probs.append(q)

    # Compute expected cost.
    # Cf. pyro.infer.util.Dice.compute_expectation()
    # https://github.com/pyro-ppl/pyro/blob/0.3.0/pyro/infer/util.py#L212
    elbo = Expectation(tuple(log_probs),
                       tuple(costs),
                       sum_vars=sum_vars,
                       prod_vars=plates)

    loss = -elbo
    assert isinstance(loss, funsor.torch.Tensor), loss.pretty()
    return loss


# This is a wrapper for compatibility with full Pyro.
def Trace_ELBO(*args, **kwargs):
    return elbo


# This is a wrapper for compatibility with full Pyro.
def TraceMeanField_ELBO(*args, **kwargs):
    # TODO Use exact KLs where possible.
    return elbo
