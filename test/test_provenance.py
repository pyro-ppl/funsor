# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from funsor.terms import to_funsor
from funsor.torch.provenance import ProvenanceTensor


@pytest.mark.parametrize(
    "data1,provenance1",
    [
        (torch.tensor([1]), frozenset({"a"})),
    ],
)
@pytest.mark.parametrize(
    "data2,provenance2",
    [
        (torch.tensor([2]), frozenset({"b"})),
        (torch.tensor([2]), None),
        (2, None),
    ],
)
def test_provenance(data1, provenance1, data2, provenance2):
    #  breakpoint()
    #  mo = MyObject(data1, extra_data=provenance1)
    if provenance1 is not None:
        data1 = ProvenanceTensor(data1, provenance1)
    if provenance2 is not None:
        data2 = ProvenanceTensor(data2, provenance2)
    breakpoint()
    to_funsor(data1)

    expected = frozenset.union(
        *[m for m in (provenance1, provenance2) if m is not None]
    )
    actual = torch.add(data1, data2)._provenance
    assert actual == expected
