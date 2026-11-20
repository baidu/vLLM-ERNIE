# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

from vllm_ernie.models.ernie5_5.nvidia import flashindexer
from vllm_ernie.models.ernie5_5.nvidia.flashindexer import (
    FLASHINDEXER_TOPK_HEADS,
    assert_flashindexer_params_loaded,
    attach_flashindexer_gate,
    flashindexer_select_heads_fused,
)

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="The ERNIE FlashIndexer gate requires CUDA"
)
INDEXER_PREFIX = "model.layers.7.attn.indexer."


def _indexer(n_head: int = 64, hidden_size: int = 4096) -> torch.nn.Module:
    module = torch.nn.Module()
    module.n_head = n_head
    module.hidden_size = hidden_size
    # The real indexer starts at the FlashIndexer-off placeholder, which is what
    # the forward tests below exercise.
    module.linear_gate = None
    return module


@requires_cuda
def test_attach_flashindexer_gate_builds_the_checkpoint_params():
    indexer = _indexer()
    attach_flashindexer_gate(indexer)
    params = dict(indexer.named_parameters())
    assert params["linear_gate.weight"].shape == (64, 4096)
    assert params["flashindexer_bias"].shape == (64,)
    assert params["flashindexer_bias"].dtype == torch.float32
    # Both names must match the checkpoint tensors the loader maps.
    assert set(params) == {"linear_gate.weight", "flashindexer_bias"}
    # The gate replaces the placeholder, so `linear_gate is None` reads it.
    assert indexer.linear_gate is indexer._modules["linear_gate"]


@requires_cuda
@pytest.mark.parametrize(
    "n_head,hidden_size", [(4, 4096), (48, 4096), (64, 100), (64, 2048 + 32)]
)
def test_attach_flashindexer_gate_rejects_shapes_the_kernel_cannot_tile(
    n_head, hidden_size
):
    with pytest.raises(ValueError, match="power-of-two head count"):
        attach_flashindexer_gate(_indexer(n_head, hidden_size))


def test_chunk_hidden_maps_a_chunk_back_onto_its_rows():
    weights = torch.zeros(12, 8)
    hidden = torch.arange(12 * 4, dtype=torch.float32).view(12, 4)
    inputs = flashindexer._FlashIndexerInputs(
        gate_weight=weights, bias=weights, hidden=hidden, weights=weights
    )
    assert torch.equal(flashindexer._chunk_hidden(inputs, weights[4:9]), hidden[4:9])
    assert torch.equal(flashindexer._chunk_hidden(inputs, weights), hidden)
    # A view the prefill loop would never produce must not be guessed at.
    with pytest.raises(RuntimeError, match="no longer"):
        flashindexer._chunk_hidden(inputs, weights[:, :4])


def test_patched_logits_kernel_passes_other_models_through():
    calls = []
    original = flashindexer._orig_mqa_logits
    flashindexer._orig_mqa_logits = lambda *args, **kwargs: calls.append((args, kwargs))
    try:
        assert flashindexer._inputs is None
        flashindexer._flashindexer_fp8_fp4_mqa_logits("q", "kv", "w", "ks", "ke", False)
    finally:
        flashindexer._orig_mqa_logits = original
    assert calls == [(("q", "kv", "w", "ks", "ke"), {"clean_logits": False})]


def test_a_missing_checkpoint_gate_is_not_served_random_init():
    params = [f"{INDEXER_PREFIX}linear_gate.weight", f"{INDEXER_PREFIX}wq_b.weight"]
    # The FP8 config switches off vLLM's own check, so this one has to fire.
    with pytest.raises(RuntimeError, match="random-init gate"):
        assert_flashindexer_params_loaded(params, {params[1]})
    assert_flashindexer_params_loaded(params, set(params))
    # FlashIndexer off: nothing was built, so nothing is missing.
    assert_flashindexer_params_loaded(params[1:], set())


def _reference(q, weights, gate_weight, bias, hidden, topk, q_sf):
    gates = hidden.float() @ gate_weight.float().t()
    top = (gates + bias.float()).topk(topk + 1, dim=-1)
    heads = top.indices[:, :topk]
    # Two GEMMs of the same gate differ in the last bits, so a row only has one
    # right answer where its top-(k+1) scores are all clearly apart.
    clear = ((top.values[:, :-1] - top.values[:, 1:]) > 1e-3).all(dim=-1)
    q_bytes = q.view(torch.uint8).view(q.shape[0], q.shape[1], -1)
    return (
        clear,
        torch.take_along_dim(q_bytes, heads[:, :, None], dim=1),
        None if q_sf is None else torch.gather(q_sf, 1, heads),
        torch.gather(weights, 1, heads) * torch.gather(gates, 1, heads),
    )


@requires_cuda
@pytest.mark.parametrize("q_dtype", [torch.float8_e4m3fn, torch.uint8])
@pytest.mark.parametrize(
    "tokens,n_head,hidden_size,topk",
    # One shape the kernel tiles in a single block, then ERNIE 5.5 Flash's own.
    [(40, 16, 256, 4), (4096, 64, 4096, FLASHINDEXER_TOPK_HEADS)],
)
def test_fused_selection_matches_a_torch_reference(
    q_dtype, tokens, n_head, hidden_size, topk
):
    torch.manual_seed(0)
    # MXFP4 Q is half as wide as FP8 Q and carries its block scales beside it.
    mxfp4 = q_dtype is torch.uint8
    head_dim = 64 if mxfp4 else 128
    kwargs = {"device": "cuda", "dtype": torch.bfloat16}
    hidden = torch.randn(tokens, hidden_size, **kwargs)
    gate_weight = torch.randn(n_head, hidden_size, **kwargs) / hidden_size**0.5
    bias = torch.randn(n_head, device="cuda")
    weights = torch.randn(tokens, n_head, device="cuda")
    if mxfp4:
        q = torch.randint(
            0, 255, (tokens, n_head, head_dim), device="cuda", dtype=torch.uint8
        )
        q_sf = torch.randint(0, 2**30, (tokens, n_head), device="cuda").int()
    else:
        q = torch.randn(tokens, n_head, head_dim, device="cuda").to(q_dtype)
        q_sf = None

    q_sel, q_sf_sel, w_sel = flashindexer_select_heads_fused(
        q, weights, gate_weight, bias, hidden, q_sf, topk_heads=topk
    )
    want_clear, want_q, want_sf, want_w = _reference(
        q, weights, gate_weight, bias, hidden, topk, q_sf
    )
    assert q_sel.shape == (tokens, topk, head_dim)
    assert q_sel.dtype == q_dtype
    # Most rows must be comparable, or the tolerance above is hiding a real break.
    assert want_clear.float().mean() > 0.9
    # The Q rows are gathered as opaque bytes, so they must match exactly.
    got_q = q_sel.view(torch.uint8).view(tokens, topk, -1)
    assert torch.equal(got_q[want_clear], want_q[want_clear])
    torch.testing.assert_close(
        w_sel[want_clear], want_w[want_clear], rtol=2e-2, atol=1e-3
    )
    if mxfp4:
        assert torch.equal(q_sf_sel[want_clear], want_sf[want_clear])
    else:
        assert q_sf_sel is None


@requires_cuda
def test_fused_selection_keeps_every_head_when_topk_is_the_head_count():
    torch.manual_seed(0)
    hidden = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(FLASHINDEXER_TOPK_HEADS, 64, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(8, FLASHINDEXER_TOPK_HEADS, device="cuda")
    q = torch.randn(8, FLASHINDEXER_TOPK_HEADS, 128, device="cuda").to(
        torch.float8_e4m3fn
    )
    _, _, w_sel = flashindexer_select_heads_fused(
        q, weights, gate, torch.zeros(FLASHINDEXER_TOPK_HEADS, device="cuda"), hidden
    )
    gates = hidden.float() @ gate.float().t()
    torch.testing.assert_close(
        w_sel.sort(dim=-1).values,
        (weights * gates).sort(dim=-1).values,
        rtol=2e-2,
        atol=1e-3,
    )
