# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch


class ProvenanceTensor(torch.Tensor):
    def __new__(cls, data, provenance=frozenset(), **kwargs):
        if not provenance:
            return data
        instance = torch.Tensor.__new__(cls)
        instance.__init__(data, provenance)
        return instance

    def __init__(self, data, provenance=frozenset()):
        assert isinstance(provenance, frozenset)
        if isinstance(data, ProvenanceTensor):
            provenance |= data._provenance
            data = data._t
        self._t = data
        self._provenance = provenance

    def __repr__(self):
        return "Provenance:\n{}\nTensor:\n{}".format(self._provenance, self._t)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        provenance = frozenset()
        _args = []
        for arg in args:
            if isinstance(arg, ProvenanceTensor):
                provenance |= arg._provenance
                _args.append(arg._t)
            elif isinstance(arg, tuple):
                _arg = []
                for a in arg:
                    if isinstance(a, ProvenanceTensor):
                        provenance |= a._provenance
                        _arg.append(a._t)
                    else:
                        _arg.append(a)
                _args.append(tuple(_arg))
            else:
                _args.append(arg)
        ret = func(*_args, **kwargs)
        if isinstance(ret, torch.Tensor):
            if provenance:
                return ProvenanceTensor(ret, provenance=provenance)
            return ret
        if isinstance(ret, tuple):
            _ret = []
            for r in ret:
                if isinstance(r, torch.Tensor) and provenance:
                    _ret.append(ProvenanceTensor(r, provenance=provenance))
                else:
                    _ret.append(r)
            return tuple(_ret)
        return ret
