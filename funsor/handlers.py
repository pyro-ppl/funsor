from __future__ import absolute_import, division, print_function

import functools

from multipledispatch import Dispatcher, dispatch
from six import add_metaclass


class Message(dict):
    # TODO use defaultdict

    _fields = ("name", "fn", "args", "kwargs", "value", "stop")

    def __init__(self, **fields):
        super(Message, self).__init__(**fields)
        for field in self._fields:
            if field not in self:
                self[field] = None


class FunsorOp(Message):
    pass


HANDLER_STACK = []
STACK_POINTER = {"ptr": -1}


def set_default_handlers(*args):
    assert not args or all(isinstance(arg, Handler) for arg in args)
    while HANDLER_STACK:
        HANDLER_STACK[-1].__exit__(None, None, None)
    for arg in args:
        arg.__enter__()


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


class OpRegistryMeta(type):
    def __init__(cls, name, bases, dct):
        super(OpRegistryMeta, cls).__init__(name, bases, dct)
        cls.dispatcher = Dispatcher(cls.__name__)


@add_metaclass(OpRegistryMeta)
class OpRegistry(Handler):
    """
    Handler with convenient op registry functionality
    """

    @dispatch(object)
    def process(self, msg):
        return super(OpRegistry, self).process(msg)

    @dispatch(FunsorOp)
    def process(self, msg):
        impl = self.dispatcher.dispatch(msg["label"])
        if impl is not None:
            msg["value"] = impl(*msg["args"], **msg["kwargs"])
        return msg

    @classmethod
    def register(cls, *term_types, **kwargs):
        return cls.dispatcher.register(tuple(term_types))


def apply_stack(msg):

    for pointer, handler in enumerate(reversed(HANDLER_STACK)):
        STACK_POINTER["ptr"] -= 1
        handler.process(msg)
        if msg["stop"]:
            break

    if msg["value"] is None:
        msg["value"] = msg["fn"](*msg["args"], **msg["kwargs"])

    for handler in HANDLER_STACK[-pointer-1:]:
        STACK_POINTER["ptr"] += 1
        handler.postprocess(msg)

    return msg


def effectful(term_type, fn=None):

    if fn is None:
        return functools.partial(effectful, term_type)

    term_label = None
    if not issubclass(term_type, Message):
        # XXX hack to make OpRegistry work
        term_label = term_type
        term_type = FunsorOp

    assert issubclass(term_type, Message)

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

            value = apply_stack(initial_msg)["value"]

        return value

    return _fn
