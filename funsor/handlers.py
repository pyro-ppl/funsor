from __future__ import absolute_import, division, print_function

import functools
import contextlib

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
        if self["label"] is None:
            self["label"] = self["fn"]


HANDLER_STACK = []
STACK_POINTER = {"ptr": -1}


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
            v = self.fn(*args, **kwargs)
            return v


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


def apply_stack(msg, stack=None):

    if stack is None:
        stack = HANDLER_STACK[:]

    for pointer, handler in enumerate(reversed(stack)):
        STACK_POINTER["ptr"] -= 1
        handler.process(msg)
        if msg["stop"]:
            break

    if msg["value"] is None:
        msg["value"] = msg["fn"](*msg["args"], **msg["kwargs"])

    for handler in stack[-pointer-1:]:
        STACK_POINTER["ptr"] += 1
        handler.postprocess(msg)

    return msg


@contextlib.contextmanager
def stack_swap():
    """a bit of gross logic for multiprompt handlers"""
    # TODO put into apply_stack
    # TODO make more efficient
    # print("BEFORE", term_label, HANDLER_STACK, prev_stack, prev_pointer)
    prev_pointer = STACK_POINTER["ptr"]
    prev_stack = HANDLER_STACK[:]
    STACK_POINTER["ptr"] = -1
    HANDLER_STACK.clear()
    HANDLER_STACK.extend(prev_stack[:len(prev_stack) + prev_pointer + 1])
    yield
    STACK_POINTER["ptr"] = prev_pointer  # TODO put into apply_stack
    HANDLER_STACK.clear()
    HANDLER_STACK.extend(prev_stack)
    # print("AFTER", term_label, HANDLER_STACK, prev_stack, prev_pointer)


def effectful(term_type, fn=None):

    term_label = None
    if not issubclass(term_type, Message):
        # XXX hack to make OpRegistry work
        term_label = term_type
        term_type = FunsorOp

    assert issubclass(term_type, Message)

    if fn is None:
        return functools.partial(effectful, term_type)

    @stack_swap()
    def _fn(*args, **kwargs):

        if not HANDLER_STACK:
            value = fn(*args, **kwargs)
        else:
            initial_msg = term_type(
                name=kwargs.pop("name", None),
                fn=fn,
                args=args,
                kwargs=kwargs,
                value=None,
                label=term_label,
            )

            value = apply_stack(initial_msg, stack=HANDLER_STACK)["value"]

        return value

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
