# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ERNIE FlashIndexer: gate params, prefill head selection, fused kernel."""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Container, Iterable, Iterator

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

# Checkpoint param names, relative to the indexer module, as the ckpt spells them.
FLASHINDEXER_GATE_PARAM_SUFFIX = "linear_gate.weight"
FLASHINDEXER_BIAS_PARAM_SUFFIX = "flashindexer_bias"
FLASHINDEXER_PARAM_SUFFIXES = (
    FLASHINDEXER_GATE_PARAM_SUFFIX,
    FLASHINDEXER_BIAS_PARAM_SUFFIX,
)

# Checkpoint subfolder holding the gate tensors, outside the main weight index.
FLASHINDEXER_WEIGHTS_SUBFOLDER = "flashindexer"

# Heads kept per token; 8 is a native DeepGEMM head count, so no head padding.
FLASHINDEXER_TOPK_HEADS = 8
_BLOCK_M = 32
_BLOCK_K = 64


def attach_flashindexer_gate(indexer: nn.Module) -> None:
    """Give a DSA indexer the learned FlashIndexer gate and its load-time bias."""
    n_head, hidden_size = indexer.n_head, indexer.hidden_size
    if not current_platform.is_cuda():
        raise ValueError("enable_flashindexer needs CUDA with the DeepGEMM indexer.")
    # The kernel spans all heads with one tl.arange and tiles hidden by _BLOCK_K.
    if (
        n_head < FLASHINDEXER_TOPK_HEADS
        or n_head & (n_head - 1)
        or hidden_size % _BLOCK_K
    ):
        raise ValueError(
            f"FlashIndexer needs a power-of-two head count >= "
            f"{FLASHINDEXER_TOPK_HEADS} and hidden size divisible by {_BLOCK_K}, "
            f"got index_n_heads={n_head}, hidden_size={hidden_size}."
        )
    # Replicated like the indexer, in the model dtype, loaded by the default loader.
    indexer.linear_gate = nn.Linear(hidden_size, n_head, bias=False)
    indexer.flashindexer_bias = nn.Parameter(
        torch.zeros(n_head, dtype=torch.float32), requires_grad=False
    )
    # Patched here, not in patches.py, which runs in every vLLM process.
    patch_sparse_attn_indexer_flashindexer()
    logger.info_once(
        "FlashIndexer prefill head selection: top %d of %d heads.",
        FLASHINDEXER_TOPK_HEADS,
        n_head,
    )


def assert_flashindexer_params_loaded(
    params: Iterable[str], loaded: Container[str]
) -> None:
    """Raise if the gate was built but the subfolder carried nothing to fill it."""
    # An FP8 config turns off vLLM's own unloaded-param check, so without this a
    # gate-less checkpoint would serve a random-init gate and never say so.
    missing = sorted(n for n in params if n.endswith(FLASHINDEXER_PARAM_SUFFIXES))
    missing = [n for n in missing if n not in loaded]
    if missing:
        raise RuntimeError(
            f"enable_flashindexer is on, but {FLASHINDEXER_WEIGHTS_SUBFOLDER}/ did "
            f"not fill {len(missing)} gate parameters, which would score prefill "
            f"with a random-init gate: {missing[:4]}. Drop the "
            "`--hf-overrides '{\"enable_flashindexer\": true}'` to run all heads."
        )


@triton.jit
def _flashindexer_fused_gemm_kernel(
    hidden_ptr,  # [B, HIDDEN] bf16
    weight_ptr,  # [NH, HIDDEN] bf16 (Linear weight, row = head)
    bias_ptr,  # [NH] fp32
    w_gate_ptr,  # [B, NH] fp32 (per-row per-head indexer weights)
    q_i32_ptr,  # [B, NH, HDI32] int32 (Q bytes reinterpreted)
    q_sf_ptr,  # [B, NH] int32 (packed Q block scales; MXFP4 only)
    q_sel_i32_ptr,  # [B, TOPK, HDI32] int32 (out)
    q_sf_sel_ptr,  # [B, TOPK] int32 (out; MXFP4 only)
    w_sel_ptr,  # [B, TOPK] fp32 (out)
    B,
    HIDDEN: tl.constexpr,
    NH: tl.constexpr,
    TOPK: tl.constexpr,
    HDI32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_Q_SF: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    m_mask = offs_m < B
    offs_n = tl.arange(0, NH)  # [NH]

    # (a) Gate GEMM: gates[BM, NH] = hidden[BM, K] @ weight^T[K, NH].
    acc = tl.zeros((BLOCK_M, NH), dtype=tl.float32)
    for k0 in range(0, HIDDEN, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)  # [BK]
        h_tile = tl.load(
            hidden_ptr + offs_m[:, None] * HIDDEN + offs_k[None, :],
            mask=m_mask[:, None],
            other=0.0,
        )  # [BM,BK] bf16
        # weight[n,k] -> need [BK,NH]: w_tile[k,n] = weight_ptr[n*HIDDEN + k]
        w_tile = tl.load(
            weight_ptr + offs_n[None, :] * HIDDEN + offs_k[:, None]
        )  # [BK,NH] bf16
        acc = tl.dot(h_tile, w_tile, acc)  # [BM,NH]

    # (b) Raw gate logits, no activation: rank by gate + bias, reweight by the gate.
    gates = acc  # [BM,NH]
    scores = gates + tl.load(bias_ptr + offs_n).to(tl.float32)[None, :]
    # PDL: overlap the gate GEMM with the predecessor kernel, then wait for it
    # before the first load of Q / weights, which it may still be writing.
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
    w_vec = tl.load(
        w_gate_ptr + offs_m[:, None] * NH + offs_n[None, :],
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)  # [BM,NH]

    # (c) TOPK passes of argmax over heads, batched over the BM rows.
    d = tl.arange(0, HDI32)  # [HDI32]
    NEG_INF = -float("inf")
    cur = scores
    for i in range(TOPK):
        idx = tl.argmax(cur, axis=1).to(tl.int32)  # [BM] i-th largest head
        sel_mask = offs_n[None, :] == idx[:, None]  # [BM,NH]
        cur = tl.where(sel_mask, NEG_INF, cur)
        gate_at = tl.sum(tl.where(sel_mask, gates, 0.0), axis=1)  # [BM]
        w_at = tl.sum(tl.where(sel_mask, w_vec, 0.0), axis=1)  # [BM]
        tl.store(w_sel_ptr + offs_m * TOPK + i, w_at * gate_at, mask=m_mask)
        # (d) Gather the picked head's Q as one [BM, HDI32] int32 row per pass.
        q_off = offs_m[:, None] * NH * HDI32 + idx[:, None] * HDI32 + d[None, :]
        q_vec = tl.load(q_i32_ptr + q_off, mask=m_mask[:, None], other=0)  # [BM,HDI32]
        out_off = offs_m[:, None] * TOPK * HDI32 + i * HDI32 + d[None, :]
        tl.store(q_sel_i32_ptr + out_off, q_vec, mask=m_mask[:, None])
        if HAS_Q_SF:
            # MXFP4 Q keeps its block scales beside Q, one word per head.
            sf = tl.load(q_sf_ptr + offs_m * NH + idx, mask=m_mask, other=0)
            tl.store(q_sf_sel_ptr + offs_m * TOPK + i, sf, mask=m_mask)

    # PDL: all stores are done, so let the downstream logits kernel launch.
    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()


def flashindexer_select_heads_fused(
    q: torch.Tensor,
    weights: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    hidden: torch.Tensor,
    q_sf: torch.Tensor | None = None,
    topk_heads: int = FLASHINDEXER_TOPK_HEADS,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Gate GEMM, top-k and Q gather in one kernel; Q [T, NH, D], weights [T, NH]."""
    tokens, num_heads, head_dim = q.shape
    assert weights.shape == (tokens, num_heads), tuple(weights.shape)
    assert hidden.shape[0] == tokens, tuple(hidden.shape)
    assert hidden.dtype == gate_weight.dtype, (hidden.dtype, gate_weight.dtype)
    # Q rows are gathered as opaque bytes: FP8 hands over 128 values, MXFP4 64 bytes.
    head_dim_bytes = head_dim * q.element_size()
    assert head_dim_bytes % 4 == 0, head_dim_bytes
    head_dim_i32 = head_dim_bytes // 4
    q_sel_i32 = q.new_empty((tokens, topk_heads, head_dim_i32), dtype=torch.int32)
    w_sel = weights.new_empty((tokens, topk_heads))
    q_i32 = q.contiguous().view(torch.int32)
    if q_sf is None:
        q_sf_sel = None
        q_sf_in, q_sf_out = q_i32, q_sel_i32
    else:
        assert q_sf.shape == (tokens, num_heads), tuple(q_sf.shape)
        q_sf_sel = q_sf.new_empty((tokens, topk_heads))
        q_sf_in, q_sf_out = q_sf.contiguous(), q_sf_sel

    # PDL: the gate GEMM reads neither, so it can run before the predecessor ends.
    pdl = current_platform.is_arch_support_pdl()
    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if pdl else {"USE_GDC": False}
    _flashindexer_fused_gemm_kernel[(triton.cdiv(tokens, _BLOCK_M),)](
        hidden.contiguous(),
        gate_weight.contiguous(),
        bias.contiguous(),
        weights.contiguous(),
        q_i32,
        q_sf_in,
        q_sel_i32,
        q_sf_out,
        w_sel,
        tokens,
        HIDDEN=hidden.shape[-1],
        NH=num_heads,
        TOPK=topk_heads,
        HDI32=head_dim_i32,
        BLOCK_M=_BLOCK_M,
        BLOCK_K=_BLOCK_K,
        HAS_Q_SF=q_sf is not None,
        **pdl_kwargs,
    )
    return q_sel_i32.view(q.dtype), q_sf_sel, w_sel


@dataclasses.dataclass(frozen=True)
class _FlashIndexerInputs:
    """One indexer forward's gate inputs, for the patched prefill kernel."""

    gate_weight: torch.Tensor
    bias: torch.Tensor
    hidden: torch.Tensor
    weights: torch.Tensor


def _chunk_hidden(
    inputs: _FlashIndexerInputs, weights_slice: torch.Tensor
) -> torch.Tensor:
    """The gate input rows behind a chunk's `weights[row_start:row_end]` view."""
    full = inputs.weights
    row_bytes = full.stride(0) * full.element_size()
    offset = weights_slice.data_ptr() - full.data_ptr()
    rows = weights_slice.shape[0]
    if (
        weights_slice.shape[1:] != full.shape[1:]
        or row_bytes <= 0
        or offset < 0
        or offset % row_bytes
        or offset // row_bytes + rows > full.shape[0]
    ):
        raise RuntimeError(
            "ERNIE FlashIndexer cannot map the indexer chunk back onto its Q rows; "
            "vllm.model_executor.layers.sparse_attn_indexer no longer "
            "slices `weights` row-wise for the prefill logits kernel."
        )
    start = offset // row_bytes
    return inputs.hidden[start : start + rows]


_inputs: _FlashIndexerInputs | None = None
_orig_mqa_logits = None


@contextlib.contextmanager
def flashindexer_context(
    indexer: nn.Module, hidden: torch.Tensor, weights: torch.Tensor
) -> Iterator[None]:
    """Publish one indexer's gate inputs to the patched prefill logits kernel."""
    global _inputs
    previous = _inputs
    _inputs = _FlashIndexerInputs(
        gate_weight=indexer.linear_gate.weight,
        bias=indexer.flashindexer_bias,
        hidden=hidden,
        weights=weights,
    )
    try:
        yield
    finally:
        _inputs = previous


def _flashindexer_fp8_fp4_mqa_logits(
    q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits
):
    """`fp8_fp4_mqa_logits` narrowed to the FlashIndexer heads of this chunk's rows."""
    inputs = _inputs
    if inputs is None:
        # Another model's indexer, or FlashIndexer off.
        return _orig_mqa_logits(
            q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=clean_logits
        )
    q_values, q_sf = q
    # Proof that the selection ran, not just that the gate was built.
    logger.info_once(
        "FlashIndexer selecting %d of %d heads for the prefill logits kernel.",
        FLASHINDEXER_TOPK_HEADS,
        q_values.shape[1],
    )
    q_sel, q_sf_sel, w_sel = flashindexer_select_heads_fused(
        q_values,
        weights,
        inputs.gate_weight,
        inputs.bias,
        _chunk_hidden(inputs, weights),
        q_sf,
    )
    return _orig_mqa_logits(
        (q_sel, q_sf_sel),
        kv,
        w_sel,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=clean_logits,
    )


def patch_sparse_attn_indexer_flashindexer() -> None:
    """Route the ragged prefill logits kernel through the head selection."""
    global _orig_mqa_logits
    from vllm.model_executor.layers import sparse_attn_indexer

    current = sparse_attn_indexer.fp8_fp4_mqa_logits
    if getattr(current, "_ernie5_flashindexer", False):
        return
    _orig_mqa_logits = current
    _flashindexer_fp8_fp4_mqa_logits._ernie5_flashindexer = True  # type: ignore[attr-defined]
    sparse_attn_indexer.fp8_fp4_mqa_logits = _flashindexer_fp8_fp4_mqa_logits
