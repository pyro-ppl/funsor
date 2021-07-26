# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch


class MetadataTensor(object):
    def __init__(self, data, metadata=frozenset(), **kwargs):
        assert isinstance(metadata, frozenset)
        self._t = torch.as_tensor(data, **kwargs)
        self._metadata = metadata

    def __repr__(self):
        return "Metadata:\n{}\n\ndata:\n{}".format(self._metadata, self._t)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        meta = frozenset().union(
            *tuple(a._metadata for a in args if hasattr(a, "_metadata"))
        )
        args = [a._t if hasattr(a, "_t") else a for a in args]
        ret = func(*args, **kwargs)
        return MetadataTensor(ret, metadata=meta)
