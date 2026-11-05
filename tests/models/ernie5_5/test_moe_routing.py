# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

from vllm_ernie.models.ernie5_5.nvidia.moe import (
    _ernie5_5_routing_torch,
    ernie5_5_routing,
    split_sigmoid_sum,
)

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")


def _router_logits(num_tokens):
    logits = torch.randn(num_tokens, 1024, device="cuda", dtype=torch.float32) * 4
    if num_tokens:
        # Saturated, denormal-result and infinite sigmoid inputs.
        edges = torch.tensor(
            [float("inf"), -float("inf"), 0.0, -0.0, 30.0, -88.0, -95.0, -104.0],
            device="cuda",
        )
        logits[0, : edges.numel()] = edges
        logits[-1, 512 : 512 + edges.numel()] = edges.flip(0)
        # Both halves in (-88.7, -87.3): denormal sigmoids that torch keeps.
        tiny = torch.tensor([-88.0, -87.5, -88.5, -float("inf")], device="cuda")
        logits[:, 16:20] = tiny
        logits[:, 528:532] = tiny.roll(1)
    return logits


@pytest.mark.parametrize("num_tokens", [0, 1, 7, 256, 65536])
def test_split_sigmoid_sum(num_tokens):
    """Bitwise torch's ``sigmoid(l0) + sigmoid(l1)``."""
    set_random_seed(0)
    logits = _router_logits(num_tokens)
    ref = logits[:, :512].sigmoid() + logits[:, 512:].sigmoid()
    assert torch.equal(split_sigmoid_sum(logits, 512), ref)


def test_split_sigmoid_sum_strided():
    set_random_seed(0)
    logits = (torch.randn(7, 2048, device="cuda", dtype=torch.float32) * 4)[:, ::2]
    ref = logits[:, :512].sigmoid() + logits[:, 512:].sigmoid()
    torch.testing.assert_close(split_sigmoid_sum(logits, 512), ref)


@pytest.mark.parametrize("num_tokens", [0, 1, 7, 256, 65536])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("topk", [10, 8, 3])
@pytest.mark.parametrize("ties", [False, True])
def test_fused_routing_matches_torch(num_tokens, renormalize, topk, ties):
    """Bitwise the torch routing, ids in the same torch.topk(sorted=False) slots."""
    set_random_seed(0)
    scores = split_sigmoid_sum(_router_logits(num_tokens), 512)
    bias = torch.randn(512, device="cuda") * 0.1
    if ties:
        # Many equal biased scores, also at the k-th value.
        scores = (scores * 4).round() / 4
        bias = (bias * 8).round() / 8
    scale = torch.rand(512, device="cuda") + 2.5
    weights, ids = ernie5_5_routing(
        None, scores, topk, renormalize, bias=bias, scale=scale
    )
    ref_w, ref_ids = _ernie5_5_routing_torch(scores, topk, renormalize, bias, scale)
    assert ids.dtype == torch.int32 and weights.dtype == torch.float32
    assert torch.equal(ids, ref_ids)
    assert torch.equal(weights, ref_w)
