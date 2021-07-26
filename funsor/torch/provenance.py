# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch


class ProvenanceTensor(torch.Tensor):
    def __new__(cls, data, provenance=frozenset(), **kwargs):
        assert isinstance(provenance, frozenset)
        t = torch.Tensor._make_subclass(cls, data)
        t._provenance = provenance
        return t

    def __repr__(self):
        return "Provenance:\n{}\nTensor:\n{}".format(
            self._provenance, torch.Tensor._make_subclass(torch.Tensor, self)
        )

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        provenance = frozenset()
        _args = []
        for arg in args:
            if isinstance(arg, ProvenanceTensor):
                provenance |= arg._provenance
                _args.append(torch.Tensor._make_subclass(torch.Tensor, arg))
            else:
                _args.append(arg)
        ret = func(*_args, **kwargs)
        if isinstance(ret, torch.Tensor):
            return ProvenanceTensor(ret, provenance=provenance)
        return ret
