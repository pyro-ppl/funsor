# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch


class MetadataTensor(torch.Tensor):
    def __new__(cls, data, metadata=frozenset(), **kwargs):
        assert isinstance(metadata, frozenset)
        t = torch.Tensor._make_subclass(cls, data)
        t._metadata = metadata
        return t

    def __repr__(self):
        return "Metadata:\n{}\ndata:\n{}".format(
            self._metadata, torch.Tensor._make_subclass(torch.Tensor, self)
        )

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        meta = frozenset()
        _args = []
        for arg in args:
            if isinstance(arg, MetadataTensor):
                meta |= arg._metadata
                _args.append(torch.Tensor._make_subclass(torch.Tensor, arg))
            else:
                _args.append(arg)
        ret = func(*_args, **kwargs)
        return MetadataTensor(ret, metadata=meta)
