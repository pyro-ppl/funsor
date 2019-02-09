from __future__ import absolute_import, division, print_function

import opt_einsum

import funsor.ops as ops
from funsor.terms import Binary, Finitary, Funsor, Reduction, Tensor, Unitary

from .paths import greedy


HANDLER_STACK = []

class Messenger(object):
    def __init__(self, fn=None):
        self.fn = fn

    def __enter__(self):
        HANDLER_STACK.append(self)

    def __exit__(self, *args, **kwargs):
        assert HANDLER_STACK[-1] is self
        HANDLER_STACK.pop()
    
    def process_message(self, msg):
        pass

    def postprocess_message(self, msg):
        pass

    def __call__(self, *args, **kwargs):
        with self:
            return self.fn(*args, **kwargs)


class EvalPass(Messenger):

    _terms_handled = {}

    def process_message(self, msg):
        if msg["type"] in self._terms_handled:
            msg["value"] = self._terms_handled[msg["type"]](
                *msg["args"], **msg["kwargs"])
        return msg

    @classmethod
    def register(cls, *term_types):
        def _fn(fn):
            for term_type in term_types:
                assert term_type not in cls._terms_handled, "cannot override"
                cls._terms_handled[term_type] = fn
            return fn
        return _fn


def effectful(term_type):

    assert issubclass(term_type, Funsor)

    def _wrap(*args, **kwargs):
        if not HANDLER_STACK:
            return term_type(*args, **kwargs)

        initial_msg = {
            "type": term_type,
            "fn": term_type,
            "args": args,
            "kwargs": kwargs,
            "value": None,
        }

        return apply_stack(initial_msg)["value"]

    return _wrap


def apply_stack(msg):
    for pointer, handler in enumerate(reversed(HANDLER_STACK)):
        handler.process_message(msg)
        if msg.get("stop"):
            break
    if msg["value"] is None:
        msg["value"] = msg["fn"](*msg["args"], **msg["kwargs"])

    for handler in HANDLER_STACK[-pointer-1:]:
        handler.postprocess_message(msg)
    return msg


def default_handler(handler):
    assert isinstance(handler, Messenger)
    def _wrapper(fn):
        def _fn(*args, **kwargs):
            if not HANDLER_STACK and not isinstance(fn, Messenger):
                with handler:
                    return fn(*args, **kwargs)
            return fn(*args, **kwargs)
        return _fn
    return _wrapper
