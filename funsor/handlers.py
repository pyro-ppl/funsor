from __future__ import absolute_import, division, print_function

import functools
from collections import namedtuple


# the type of a label is the op type, e.g. sample/param
Label = namedtuple("Label", ["name"])
Label.__new__.__defaults__ = (None,)


HANDLER_STACK = []


class Handler(object):
    def __init__(self, fn=None):
        self.fn = fn

    def __enter__(self):
        HANDLER_STACK.append(self)

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            assert HANDLER_STACK[-1] is self
            HANDLER_STACK.pop()
        else:
            if self in HANDLER_STACK:
                loc = HANDLER_STACK.index(self)
                for i in range(loc, len(HANDLER_STACK)):
                    HANDLER_STACK.pop()

    def process(self, msg):
        return msg

    def postprocess(self, msg):
        return msg

    def __call__(self, *args, **kwargs):
        with self:
            return self.fn(*args, **kwargs)


class OpRegistry(Handler):
    """
    Handler with convenient op registry functionality
    """

    _terms_processed = {}
    _terms_postprocessed = {}

    def process(self, msg):
        if msg["label"] in self._terms_processed:
            msg["value"] = self._terms_processed[msg["label"]](
                *msg["args"], **msg["kwargs"])
        return msg

    def postprocess(self, msg):
        if msg["label"] in self._terms_postprocessed:
            msg["value"] = self._terms_postprocessed[msg["label"]](
                *msg["args"], **msg["kwargs"])
        return msg

    @classmethod
    def register(cls, *term_types, **kwargs):
        post = kwargs.pop('post', False)
        assert not kwargs

        def _fn(fn):
            for term_type in term_types:
                if post:
                    assert term_type not in cls._terms_postprocessed, \
                        "cannot override"
                    cls._terms_postprocessed[term_type] = fn
                else:
                    assert term_type not in cls._terms_processed, \
                        "cannot override"
                    cls._terms_processed[term_type] = fn
            return fn

        return _fn


def apply_stack(msg):
    for pointer, handler in enumerate(reversed(HANDLER_STACK)):
        handler.process(msg)
        if msg.get("stop"):
            break
    if msg["value"] is None:
        msg["value"] = msg["fn"](*msg["args"], **msg["kwargs"])

    for handler in HANDLER_STACK[-pointer-1:]:
        handler.postprocess(msg)
    return msg


def effectful(term_type, fn=None):

    if fn is None:
        return functools.partial(effectful, term_type)

    def _fn(*args, **kwargs):

        if not HANDLER_STACK:
            return fn(*args, **kwargs)

        name = kwargs.pop("name", None)
        initial_msg = {
            "label": term_type,
            "name": name,
            "fn": fn,
            "args": args,
            "kwargs": kwargs,
            "value": None,
        }

        return apply_stack(initial_msg)["value"]

    return _fn


def default_handler(handler):
    """annotate a function with a default handler"""
    assert isinstance(handler, Handler)

    def _wrapper(fn):
        def _fn(*args, **kwargs):
            if not HANDLER_STACK and not isinstance(fn, Handler):
                with handler:
                    return fn(*args, **kwargs)
            return fn(*args, **kwargs)
        return _fn

    return _wrapper
