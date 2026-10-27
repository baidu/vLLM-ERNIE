# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="ERNIE 5.5 DSA requires CUDA"
)

WINDOW, TOPK = 128, 2048


def test_merge_swa_topk_appends_window():
    """The indexer writes contiguous top-k rows; attention reads wider rows."""
    from vllm_ernie5_plugin.ernie5.nvidia.attention import merge_swa_topk

    positions = torch.tensor([0, 5, 127, 128, 200, 2175, 2176, 5000], device="cuda")
    num_tokens, width = positions.shape[0], TOPK + WINDOW
    topk = torch.randint(
        0, 1 << 20, (num_tokens, TOPK), dtype=torch.int32, device="cuda"
    )
    out = torch.full((num_tokens, width), -7, dtype=torch.int32, device="cuda")

    merge_swa_topk(topk, out, positions, WINDOW)

    for row, pos in enumerate(positions.tolist()):
        seq = pos + 1
        n, w = min(max(seq - WINDOW, 0), TOPK), min(seq, WINDOW)
        expected = torch.full((width,), -1, dtype=torch.int32)
        expected[:n] = topk[row, :n].cpu()
        expected[n : n + w] = torch.arange(seq - w, seq, dtype=torch.int32)
        torch.testing.assert_close(out[row].cpu(), expected, rtol=0, atol=0)


def test_packed_block_stride_alignment_fits_every_layer_row():
    """HCA, DSA and indexer pages share one stride in the hybrid KV group."""
    from vllm_ernie5_plugin.ernie5.nvidia.attention import packed_block_stride_alignment

    hf_config = SimpleNamespace(
        head_dim=512, mla_kv_lora_rank=512, mla_qk_rope_head_dim=64
    )
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_config=hf_config))

    alignment = packed_block_stride_alignment(vllm_config)

    assert alignment % 512 == 0
    assert alignment % (512 * 2) == 0
    assert alignment % ((512 + 64) * 2) == 0


@pytest.mark.parametrize("spec", [None, SimpleNamespace()])
def test_dsa_kernel_block_sizes_accept_kv_cache_spec(spec):
    """vLLM passes the layer's KV spec when picking kernel block sizes."""
    from vllm.v1.attention.backend import MultipleOf

    from vllm_ernie5_plugin.ernie5.nvidia.attention import (
        Ernie5_5FlashMLASparseBackend,
        Ernie5_5IndexerBackend,
    )

    for backend in (Ernie5_5FlashMLASparseBackend, Ernie5_5IndexerBackend):
        assert backend.get_supported_kernel_block_sizes() == [MultipleOf(64)]
        assert backend.get_supported_kernel_block_sizes(spec) == [MultipleOf(64)]
