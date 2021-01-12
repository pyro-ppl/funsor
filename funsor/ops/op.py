# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import weakref

from multipledispatch import Dispatcher


class OpCacheMeta(type):
    """
    Metaclass for caching op instance construction.
    """
    # use a WeakKeyDictionary to allow garbage collection of out-of-scope op classes.
    # this is especially important when some ops are dynamically generated and
    # their instances are large, e.g. an op corresponding to a neural network.
    _cls_caches = weakref.WeakKeyDictionary()  # class -> {args: instance}

    def __call__(cls, *args, **kwargs):
        if cls not in OpCacheMeta._cls_caches:
            OpCacheMeta._cls_caches[cls] = {}
        cls_cache = OpCacheMeta._cls_caches[cls]
        key = tuple(args) + tuple(kwargs.items())
        if key not in cls_cache:
            cls_cache[key] = super(OpCacheMeta, cls).__call__(*args, **kwargs)
        return cls_cache[key]


class Op(Dispatcher):
    def __init__(self, fn, *, name=None):
        if isinstance(fn, str):
            fn, name = None, fn
        if name is None:
            name = fn.__name__
        super(Op, self).__init__(name)
        if fn is not None:
            # register as default operation
            for nargs in (1, 2):
                default_signature = (object,) * nargs
                self.add(default_signature, fn)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return self.__name__

    def __repr__(self):
        return "ops." + self.__name__

    def __str__(self):
        return self.__name__


class TransformOp(Op):
    def set_inv(self, fn):
        """
        :param callable fn: A function that inputs an arg ``y`` and outputs a
            value ``x`` such that ``y=self(x)``.
        """
        assert callable(fn)
        self.inv = fn
        return fn

    def set_log_abs_det_jacobian(self, fn):
        """
        :param callable fn: A function that inputs two args ``x, y``, where
            ``y=self(x)``, and returns ``log(abs(det(dy/dx)))``.
        """
        assert callable(fn)
        self.log_abs_det_jacobian = fn
        return fn

    @staticmethod
    def inv(x):
        raise NotImplementedError

    @staticmethod
    def log_abs_det_jacobian(x, y):
        raise NotImplementedError


# Op registration tables.
DISTRIBUTIVE_OPS = set()  # (add, mul) pairs
UNITS = {}                # op -> value
PRODUCT_INVERSES = {}     # op -> inverse op

__all__ = [
    'DISTRIBUTIVE_OPS',
    'Op',
    'OpCacheMeta',
    'PRODUCT_INVERSES',
    'TransformOp',
    'UNITS',
]
