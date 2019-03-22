from __future__ import absolute_import, division, print_function

import functools


class lazy_property(object):
    def __init__(self, fn):
        self.fn = fn
        functools.update_wrapper(self, fn)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        value = self.fn(obj)
        setattr(obj, self.fn.__name__, value)
        return value
