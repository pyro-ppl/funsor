# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from funsor.torch import MetadataTensor


@pytest.mark.parametrize(
    "data1,metadata1",
    [
        (torch.tensor([1]), frozenset({"a"})),
    ],
)
@pytest.mark.parametrize(
    "data2,metadata2",
    [
        (torch.tensor([2]), frozenset({"b"})),
        (torch.tensor([2]), None),
        (2, None),
    ],
)
def test_metadata(data1, metadata1, data2, metadata2):
    if metadata1 is not None:
        data1 = MetadataTensor(data1, metadata1)
    if metadata2 is not None:
        data2 = MetadataTensor(data2, metadata2)

    expected = frozenset.union(*[m for m in (metadata1, metadata2) if m is not None])
    actual = torch.add(data1, data2)._metadata
    assert actual == expected
