# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from pyro.ops.indexing import Vindex

from funsor.torch.provenance import ProvenanceTensor


@pytest.mark.parametrize("op", ["log", "exp", "long"])
@pytest.mark.parametrize(
    "data,provenance",
    [
        (torch.tensor([1]), "ab"),
        (torch.tensor([1]), "a"),
    ],
)
def test_unary(op, data, provenance):
    data = ProvenanceTensor(data, frozenset(provenance))

    expected = frozenset(provenance)
    actual = getattr(data, op)()._provenance
    assert actual == expected


@pytest.mark.parametrize("data1,provenance1", [(torch.tensor([1]), "a")])
@pytest.mark.parametrize(
    "data2,provenance2",
    [
        (torch.tensor([2]), "b"),
        (torch.tensor([2]), ""),
        (2, ""),
    ],
)
def test_binary_add(data1, provenance1, data2, provenance2):
    data1 = ProvenanceTensor(data1, frozenset(provenance1))
    if provenance2:
        data2 = ProvenanceTensor(data2, frozenset(provenance2))

    expected = frozenset(provenance1 + provenance2)
    actual = torch.add(data1, data2)._provenance
    assert actual == expected


@pytest.mark.parametrize(
    "data1,provenance1",
    [
        (torch.tensor([0, 1]), "a"),
        (torch.tensor([0, 1]), ""),
    ],
)
@pytest.mark.parametrize(
    "data2,provenance2",
    [
        (torch.tensor([0]), "b"),
        (torch.tensor([1]), ""),
    ],
)
def test_indexing(data1, provenance1, data2, provenance2):
    if provenance1:
        data1 = ProvenanceTensor(data1, frozenset(provenance1))
    if provenance2:
        data2 = ProvenanceTensor(data2, frozenset(provenance2))

    expected = frozenset(provenance1 + provenance2)
    actual = getattr(data1[data2], "_provenance", frozenset())
    assert actual == expected


@pytest.mark.parametrize(
    "data1,provenance1",
    [
        (torch.tensor([[0, 1], [2, 3]]), "a"),
        (torch.tensor([[0, 1], [2, 3]]), ""),
    ],
)
@pytest.mark.parametrize(
    "data2,provenance2",
    [
        (torch.tensor([0.0, 1.0]), "b"),
        (torch.tensor([0.0, 1.0]), ""),
    ],
)
@pytest.mark.parametrize(
    "data3,provenance3",
    [
        (torch.tensor([0, 1]), "c"),
        (torch.tensor([0, 1]), ""),
    ],
)
def test_vindex(data1, provenance1, data2, provenance2, data3, provenance3):
    if provenance1:
        data1 = ProvenanceTensor(data1, frozenset(provenance1))
    if provenance2:
        data2 = ProvenanceTensor(data2, frozenset(provenance2))
    if provenance3:
        data3 = ProvenanceTensor(data3, frozenset(provenance3))

    expected = frozenset(provenance1 + provenance2 + provenance3)
    result = Vindex(data1)[data2.long().unsqueeze(-1), data3]
    actual = getattr(result, "_provenance", frozenset())
    assert actual == expected
