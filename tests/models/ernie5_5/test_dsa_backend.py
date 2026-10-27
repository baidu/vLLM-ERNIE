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
    from vllm_ernie.models.ernie5_5.nvidia.attention import merge_swa_topk

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
    from vllm_ernie.models.ernie5_5.nvidia.attention import (
        packed_block_stride_alignment,
    )

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

    from vllm_ernie.models.ernie5_5.nvidia.attention import (
        Ernie5_5FlashMLASparseBackend,
        Ernie5_5IndexerBackend,
    )

    for backend in (Ernie5_5FlashMLASparseBackend, Ernie5_5IndexerBackend):
        assert backend.get_supported_kernel_block_sizes() == [MultipleOf(64)]
        assert backend.get_supported_kernel_block_sizes(spec) == [MultipleOf(64)]


@pytest.mark.parametrize("num_rows", [1, 2, 5, 6, 7, 64, 129, 1026])
def test_hadamard_rows_matches_dense(num_rows):
    """Row counts whose numel is not a multiple of 512 included."""
    from vllm_ernie.models.ernie5_5.nvidia.attention import hadamard_rows

    dim = 128
    h = torch.ones(1, 1)
    while h.shape[0] < dim:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    h = (h / dim**0.5).cuda()
    x = torch.randn(num_rows, dim, device="cuda", dtype=torch.bfloat16)

    out = hadamard_rows(x)

    assert out.shape == x.shape
    ref = x.float() @ h
    assert ((out.float() - ref).norm() / ref.norm()).item() < 1e-2


def test_compressor_skip_not_baked_into_cuda_graph(monkeypatch):
    """Captured graphs must keep the C128 compression the capture batch skips."""
    import dataclasses

    from vllm.models.deepseek_v4.compressor import CompressorMetadata

    from vllm_ernie.models.ernie5_5.nvidia import attention

    seen = []
    monkeypatch.setattr(
        attention.DeepseekCompressor,
        "forward",
        lambda self, *a, **kw: seen.append(metadata[name].c128_boundary),
    )
    name = "model.layers.0.attn.compressor.state_cache"
    state = CompressorMetadata(
        block_table=torch.zeros(1),
        slot_mapping=torch.zeros(1),
        block_size=8,
        c128_boundary=False,
    )
    metadata = {name: state}
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=metadata),
    )
    comp = attention.Ernie5_5Compressor.__new__(attention.Ernie5_5Compressor)
    object.__setattr__(comp, "state_cache", SimpleNamespace(prefix=name))

    for capturing in (False, True):
        metadata[name] = dataclasses.replace(state)
        monkeypatch.setattr(
            torch.cuda, "is_current_stream_capturing", lambda c=capturing: c
        )
        comp.forward()

    assert seen == [False, None]
