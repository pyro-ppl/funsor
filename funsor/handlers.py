from __future__ import absolute_import, division, print_function

import functools

from multipledispatch import dispatch


class Message(dict):
    # TODO use defaultdict

    _fields = ("name", "fn", "args", "kwargs", "value", "stop")

    def __init__(self, **fields):
        super(Message, self).__init__(**fields)
        for field in self._fields:
            if field not in self:
                self[field] = None


class FunsorOp(Message):

    def __init__(self, **fields):
        super(FunsorOp, self).__init__(**fields)
        self["label"] = self["fn"]


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

    @dispatch(Message)
    def process(self, msg):
        return msg

    @dispatch(Message)
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

    @dispatch(object)
    def process(self, msg):
        return super(OpRegistry, self).process(msg)

    @dispatch(object)
    def postprocess(self, msg):
        return super(OpRegistry, self).process(msg)

    @dispatch(FunsorOp)
    def process(self, msg):
        if msg["label"] in self._terms_processed:
            msg["value"] = self._terms_processed[msg["label"]](
                *msg["args"], **msg["kwargs"])
        return msg

    @dispatch(FunsorOp)
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
        if msg["stop"]:
            break
    if msg["value"] is None:
        msg["value"] = msg["fn"](*msg["args"], **msg["kwargs"])

    for handler in HANDLER_STACK[-pointer-1:]:
        handler.postprocess(msg)
    return msg


def effectful(term_type, fn=None):

    if not issubclass(term_type, Message):
        # XXX hack to make OpRegistry work
        term_type = FunsorOp

    assert issubclass(term_type, Message)

    if fn is None:
        return functools.partial(effectful, term_type)

    def _fn(*args, **kwargs):

        name = kwargs.pop("name", None)

        if not HANDLER_STACK:
            return fn(*args, **kwargs)

        initial_msg = term_type(
            name=name,
            fn=fn,
            args=args,
            kwargs=kwargs,
            value=None,
        )

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
