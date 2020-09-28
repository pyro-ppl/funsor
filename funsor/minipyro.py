# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

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
import functools
import warnings
import weakref
from collections import OrderedDict, namedtuple

import torch
from pyro.distributions import validation_enabled
from pyro.optim.clipped_adam import ClippedAdam as _ClippedAdam

import funsor


# Funsor repreresents distributions in a fundamentally different way from
# torch.Distributions and Pyro: funsor distributions are densities whereas
# torch Distributions are samplers. This class is a compatibility wrapper
# between the two. It is used only internally in the sample() function.
class Distribution(object):
    def __init__(self, funsor_dist, sample_inputs=None):
        assert isinstance(funsor_dist, funsor.Funsor)
        assert not sample_inputs or all(isinstance(inp.dtype, int) for inp in sample_inputs.values())
        self.funsor_dist = funsor_dist
        self.output = self.funsor_dist.inputs["value"]
        self.sample_inputs = sample_inputs

    def log_prob(self, value):
        result = self.funsor_dist(value=value)
        if self.sample_inputs:
            result = result + funsor.tensor.Tensor(
                torch.zeros(*(size.dtype for size in self.sample_inputs.values())),
                self.sample_inputs
            )
        return result

    # Draw a sample.
    def __call__(self):
        with funsor.interpretation(funsor.terms.eager):
            dist = self.funsor_dist(value='value')
            delta = dist.sample(frozenset(['value']), sample_inputs=self.sample_inputs)
        if isinstance(delta, funsor.cnf.Contraction):
            assert len(delta.terms) == 2
            assert any(isinstance(t, funsor.delta.Delta) for t in delta.terms)
            delta = [t for t in delta.terms if isinstance(t, funsor.delta.Delta)][0]
        assert isinstance(delta, funsor.delta.Delta)
        return delta.terms[0][1][0]

    # Similar to torch.distributions.Distribution.expand().
    def expand_inputs(self, name, size):
        if name in self.funsor_dist.inputs:
            assert self.funsor_dist.inputs[name] == funsor.Bint[int(size)]
            return self
        inputs = OrderedDict([(name, funsor.Bint[int(size)])])
        if self.sample_inputs:
            inputs.update(self.sample_inputs)
        return Distribution(self.funsor_dist, sample_inputs=inputs)


# Pyro keeps track of two kinds of global state:
# i)  The effect handler stack, which enables non-standard interpretations of
#     Pyro primitives like sample();
#     See http://docs.pyro.ai/en/0.3.1/poutine.html
# ii) Trainable parameters in the Pyro ParamStore;
#     See http://docs.pyro.ai/en/0.3.1/parameters.html

PYRO_STACK = []
PARAM_STORE = {}  # maps name -> (unconstrained_value, constraint)


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
        assert msg["type"] != "sample" or msg["name"] not in self.trace, \
            "sample sites must have unique names"
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


# seed is used to fix the RNG state when calling a model.
class seed(Messenger):
    def __init__(self, fn=None, rng_seed=None):
        self.rng_seed = rng_seed
        super(seed, self).__init__(fn)

    def __enter__(self):
        self.old_rng_state = torch.get_rng_state()
        torch.manual_seed(self.rng_seed)

    def __exit__(self, type, value, traceback):
        torch.set_rng_state(self.old_rng_state)


# Conditional independence is recorded as a plate context at each site.
CondIndepStackFrame = namedtuple("CondIndepStackFrame", ["name", "size", "dim"])


# This implementation of vectorized PlateMessenger broadcasts and
# records a cond_indep_stack which is later used to convert
# torch.Tensors to funsor.tensor.Tensors.
class PlateMessenger(Messenger):
    def __init__(self, fn, name, size, dim):
        assert dim < 0
        self.frame = CondIndepStackFrame(name, size, dim)
        super(PlateMessenger, self).__init__(fn)

    def process_message(self, msg):
        if msg["type"] in ("sample", "param"):
            assert self.frame.dim not in msg["cond_indep_stack"]
            msg["cond_indep_stack"][self.frame.dim] = self.frame
        if msg["type"] == "sample":
            msg["fn"] = msg["fn"].expand_inputs(self.frame.name, self.frame.size)


# This converts raw tensor.Tensors to funsor.Funsors with .inputs and .output
# based on information in msg["cond_indep_stack"] and msg["fn"].
def tensor_to_funsor(value, cond_indep_stack, output):
    assert isinstance(value, torch.Tensor)
    event_shape = output.shape
    batch_shape = value.shape[:value.dim() - len(event_shape)]
    if torch._C._get_tracing_state():
        with funsor.tensor.ignore_jit_warnings():
            batch_shape = tuple(map(int, batch_shape))
    inputs = OrderedDict()
    data = value
    for dim, size in enumerate(batch_shape):
        if size == 1:
            data = data.squeeze(dim - value.dim())
        else:
            frame = cond_indep_stack[dim - len(batch_shape)]
            assert size == frame.size, (size, frame)
            inputs[frame.name] = funsor.Bint[int(size)]
    value = funsor.tensor.Tensor(data, inputs, output.dtype)
    assert value.output == output
    return value


# The log_joint messenger is the main way of recording log probabilities.
# This is roughly the Funsor equivalent to pyro.poutine.trace.
class log_joint(Messenger):
    def __enter__(self):
        super(log_joint, self).__enter__()
        self.log_factors = OrderedDict()  # maps site name to log_prob factor
        self.plates = set()
        return self

    def process_message(self, msg):
        if msg["type"] == "sample":
            if msg["value"] is None:
                # Create a delayed sample.
                msg["value"] = funsor.Variable(msg["name"], msg["fn"].output)

    def postprocess_message(self, msg):
        if msg["type"] == "sample":
            assert msg["name"] not in self.log_factors, "all sites must have unique names"
            log_prob = msg["fn"].log_prob(msg["value"])
            self.log_factors[msg["name"]] = log_prob
            self.plates.update(f.name for f in msg["cond_indep_stack"].values())


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
    if isinstance(msg["value"], torch.Tensor):
        msg["value"] = tensor_to_funsor(msg["value"], msg["cond_indep_stack"], msg["output"])

    # A Messenger that sets msg["stop"] == True also prevents application
    # of postprocess_message by Messengers above it on the stack
    # via the pointer variable from the process_message loop
    for handler in PYRO_STACK[-pointer-1:]:
        handler.postprocess_message(msg)
    return msg


# sample is an effectful version of Distribution.sample(...)
# When any effect handlers are active, it constructs an initial message and calls apply_stack.
def sample(name, fn, obs=None, infer=None):
    # Wrap the funsor distribution in a Pyro-compatible way.
    fn = Distribution(fn)

    # if there are no active Messengers, we just draw a sample and return it as expected:
    if not PYRO_STACK:
        return fn()

    # Otherwise, we initialize a message...
    initial_msg = {
        "type": "sample",
        "name": name,
        "fn": fn,
        "args": (),
        "value": obs,
        "cond_indep_stack": {},  # maps dim to CondIndepStackFrame
        "output": fn.output,
        "infer": {} if infer is None else infer,
    }

    # ...and use apply_stack to send it to the Messengers
    msg = apply_stack(initial_msg)
    assert isinstance(msg["value"], funsor.Funsor)
    return msg["value"]


# param is an effectful version of PARAM_STORE.setdefault that also handles constraints.
# When any effect handlers are active, it constructs an initial message and calls apply_stack.
def param(name, init_value=None, constraint=torch.distributions.constraints.real, event_dim=None):
    cond_indep_stack = {}
    output = None
    if init_value is not None:
        if event_dim is None:
            event_dim = init_value.dim()
        output = funsor.Reals[init_value.shape[init_value.dim() - event_dim:]]

    def fn(init_value, constraint):
        if name in PARAM_STORE:
            unconstrained_value, constraint = PARAM_STORE[name]
        else:
            # Initialize with a constrained value.
            assert init_value is not None
            with torch.no_grad():
                constrained_value = init_value.detach()
                unconstrained_value = torch.distributions.transform_to(constraint).inv(constrained_value)
            unconstrained_value.requires_grad_()
            unconstrained_value._funsor_metadata = (cond_indep_stack, output)
            PARAM_STORE[name] = unconstrained_value, constraint

        # Transform from unconstrained space to constrained space.
        constrained_value = torch.distributions.transform_to(constraint)(unconstrained_value)
        constrained_value.unconstrained = weakref.ref(unconstrained_value)
        return tensor_to_funsor(constrained_value, *unconstrained_value._funsor_metadata)

    # if there are no active Messengers, we just draw a sample and return it as expected:
    if not PYRO_STACK:
        return fn(init_value, constraint)

    # Otherwise, we initialize a message...
    initial_msg = {
        "type": "param",
        "name": name,
        "fn": fn,
        "args": (init_value, constraint),
        "value": None,
        "cond_indep_stack": cond_indep_stack,  # maps dim to CondIndepStackFrame
        "output": output,
    }

    # ...and use apply_stack to send it to the Messengers
    msg = apply_stack(initial_msg)
    assert isinstance(msg["value"], funsor.Funsor)
    return msg["value"]


# boilerplate to match the syntax of actual pyro.plate:
def plate(name, size, dim):
    return PlateMessenger(fn=None, name=name, size=size, dim=dim)


# This is a thin wrapper around the `torch.optim.Optimizer` class that
# dynamically generates optimizers for dynamically generated parameters.
# See http://docs.pyro.ai/en/0.3.1/optimization.html
class PyroOptim(object):
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
                optim = self.TorchOptimizer([param], **self.optim_args)
                self.optim_objs[param] = optim
            # Take a gradient step for the parameter param.
            optim.step()


# We wrap some commonly used PyTorch optimizers.
class Adam(PyroOptim):
    TorchOptimizer = torch.optim.Adam


class ClippedAdam(PyroOptim):
    TorchOptimizer = _ClippedAdam


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
                loss = self.loss(self.model, self.guide, *args, **kwargs)
        # Differentiate the loss.
        funsor.to_data(loss).backward()
        # Grab all the parameters from the trace.
        params = [site["value"].data.unconstrained()
                  for site in param_capture.values()]
        # Take a step w.r.t. each parameter in params.
        self.optim(params)
        # Zero out the gradients so that they don't accumulate.
        for p in params:
            p.grad = torch.zeros_like(p.grad)
        return loss.item()


# TODO(eb8680) Replace this with funsor.Expectation.
def Expectation(log_probs, costs, sum_vars, prod_vars):
    result = 0
    for cost in costs:
        log_prob = funsor.sum_product.sum_product(
            sum_op=funsor.ops.logaddexp,
            prod_op=funsor.ops.add,
            factors=log_probs,
            plates=prod_vars,
            eliminate=(prod_vars | sum_vars) - frozenset(cost.inputs)
        )
        term = funsor.Integrate(log_prob, cost, sum_vars & frozenset(cost.inputs))
        term = term.reduce(funsor.ops.add, prod_vars & frozenset(cost.inputs))
        result += term
    return result


# This is a basic implementation of the Evidence Lower Bound, which is the
# fundamental objective in Variational Inference.
# See http://pyro.ai/examples/svi_part_i.html for details.
# This implementation uses a Dice estimator similar to TraceEnum_ELBO.
def elbo(model, guide, *args, **kwargs):
    with log_joint() as guide_log_joint:
        guide(*args, **kwargs)
    with log_joint() as model_log_joint:
        model(*args, **kwargs)

    # contract out auxiliary variables in the guide
    guide_log_probs = list(guide_log_joint.log_factors.values())
    guide_aux_vars = frozenset().union(*(f.inputs for f in guide_log_probs)) - \
        frozenset(guide_log_joint.plates) - \
        frozenset(model_log_joint.log_factors)
    if guide_aux_vars:
        guide_log_probs = funsor.sum_product.partial_sum_product(
            funsor.ops.logaddexp,
            funsor.ops.add,
            guide_log_probs,
            plates=frozenset(guide_log_joint.plates),
            eliminate=guide_aux_vars)

    # contract out auxiliary variables in the model
    model_log_probs = list(model_log_joint.log_factors.values())
    model_aux_vars = frozenset().union(*(f.inputs for f in model_log_probs)) - \
        frozenset(model_log_joint.plates) - \
        frozenset(guide_log_joint.log_factors)
    if model_aux_vars:
        model_log_probs = funsor.sum_product.partial_sum_product(
            funsor.ops.logaddexp,
            funsor.ops.add,
            model_log_probs,
            plates=frozenset(model_log_joint.plates),
            eliminate=model_aux_vars)

    # compute remaining plates and sum_dims
    plates = frozenset().union(
        *(model_log_joint.plates.intersection(f.inputs) for f in model_log_probs))
    plates = plates | frozenset().union(
        *(guide_log_joint.plates.intersection(f.inputs) for f in guide_log_probs))
    sum_vars = frozenset().union(model_log_joint.log_factors, guide_log_joint.log_factors) - \
        frozenset(model_aux_vars | guide_aux_vars)

    # Accumulate costs from model and guide and log_probs from guide.
    # Cf. pyro.infer.traceenum_elbo._compute_dice_elbo()
    # https://github.com/pyro-ppl/pyro/blob/0.3.0/pyro/infer/traceenum_elbo.py#L119
    costs = []
    log_probs = []
    for p in model_log_probs:
        costs.append(p)
    for q in guide_log_probs:
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
    assert not loss.inputs
    return loss


# Base class for elbo implementations.
class ELBO(object):
    def __init__(self, **kwargs):
        self.options = kwargs

    def __call__(self, model, guide, *args, **kwargs):
        return elbo(model, guide, *args, **kwargs)


# This is a wrapper for compatibility with full Pyro.
class Trace_ELBO(ELBO):
    def __call__(self, model, guide, *args, **kwargs):
        with funsor.interpretation(funsor.montecarlo.MonteCarlo()):
            return elbo(model, guide, *args, **kwargs)


class TraceMeanField_ELBO(ELBO):
    # TODO Use exact KLs where possible.
    pass


class TraceEnum_ELBO(ELBO):
    # TODO allow mixing of sampling and exact integration
    def __call__(self, model, guide, *args, **kwargs):
        if self.options.get("optimize", None):
            with funsor.interpretation(funsor.optimizer.optimize):
                elbo_expr = elbo(model, guide, *args, **kwargs)
            return funsor.reinterpret(elbo_expr)
        return elbo(model, guide, *args, **kwargs)


# This is a PyTorch jit wrapper that (1) delays tracing until the first
# invocation, and (2) registers pyro.param() statements with torch.jit.trace.
# This version does not support variable number of args or non-tensor kwargs.
class Jit(object):
    def __init__(self, fn, **kwargs):
        self.fn = fn
        self.ignore_jit_warnings = kwargs.get("ignore_jit_warnings", False)
        self._compiled = None
        self._param_trace = None

    def __call__(self, *args):
        # On first call, initialize params and save their names.
        if self._param_trace is None:
            with block(), trace() as tr, block(hide_fn=lambda m: m["type"] != "param"):
                self.fn(*args)
            self._param_trace = tr

        # Augment args with reads from the global param store.
        unconstrained_params = tuple(param(name).data.unconstrained()
                                     for name in self._param_trace)
        params_and_args = unconstrained_params + args

        # On first call, create a compiled elbo.
        if self._compiled is None:

            def compiled(*params_and_args):
                unconstrained_params = params_and_args[:len(self._param_trace)]
                args = params_and_args[len(self._param_trace):]
                for name, unconstrained_param in zip(self._param_trace, unconstrained_params):
                    constrained_param = param(name)  # assume param has been initialized
                    assert constrained_param.data.unconstrained() is unconstrained_param
                    self._param_trace[name]["value"] = constrained_param
                result = replay(self.fn, guide_trace=self._param_trace)(*args)
                assert not result.inputs
                assert result.output == funsor.Real
                return funsor.to_data(result)

            with validation_enabled(False), warnings.catch_warnings():
                if self.ignore_jit_warnings:
                    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                self._compiled = torch.jit.trace(compiled, params_and_args, check_trace=False)

        data = self._compiled(*params_and_args)
        return funsor.tensor.Tensor(data)


# This is a jit wrapper for ELBO implementations.
class Jit_ELBO(ELBO):
    def __init__(self, elbo, **kwargs):
        super(Jit_ELBO, self).__init__(**kwargs)
        self._elbo = elbo(**kwargs)
        self._compiled = {}  # maps (model,guide) -> Jit instances

    def __call__(self, model, guide, *args):
        if (model, guide) not in self._compiled:
            elbo = functools.partial(self._elbo, model, guide)
            self._compiled[model, guide] = Jit(elbo, **self.options)
        return self._compiled[model, guide](*args)


def JitTrace_ELBO(**kwargs):
    return Jit_ELBO(Trace_ELBO, **kwargs)


def JitTraceMeanField_ELBO(**kwargs):
    return Jit_ELBO(TraceMeanField_ELBO, **kwargs)


def JitTraceEnum_ELBO(**kwargs):
    return Jit_ELBO(TraceEnum_ELBO, **kwargs)
