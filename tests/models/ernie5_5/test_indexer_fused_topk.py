# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import dataclasses
import importlib
import importlib.util
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm import _custom_ops
from vllm.model_executor.layers import sparse_attn_indexer
from vllm.platforms import current_platform
from vllm.utils import deep_gemm
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder

from vllm_ernie.models.ernie5_5.nvidia import attention, flashindexer
from vllm_ernie.models.ernie5_5.nvidia import indexer_fused_topk as fused
from vllm_ernie.models.ernie5_5.nvidia.attention import (
    Ernie5_5Indexer,
    Ernie5_5IndexerMetadataBuilder,
)
from vllm_ernie.models.ernie5_5.nvidia.flashindexer import (
    FLASHINDEXER_TOPK_HEADS,
    flashindexer_context,
    flashindexer_select_heads_fused,
)

TOPK = 2048
HEAD_DIM = 128
WINDOW = 128
HIDDEN_SIZE = 4096
PREFIX = "model.layers.0.attn.indexer.k_cache"
VLLM_CONFIG = SimpleNamespace(
    parallel_config=SimpleNamespace(
        decode_context_parallel_size=1, prefill_context_parallel_size=1
    )
)


def _halops_unsupported_reason() -> str | None:
    if not current_platform.is_cuda():
        return "needs CUDA"
    if importlib.util.find_spec("halops") is None:
        return "HALOps is not installed"
    config = importlib.import_module(
        "halops.indexer.kernels.mqa_logits_fused_topk.config"
    )
    return config.unsupported_reason(64, HEAD_DIM, TOPK, qkv_dtype="fp8")


requires_halops_install = pytest.mark.skipif(
    importlib.util.find_spec("halops") is None, reason="HALOps is not installed"
)
requires_halops_gpu = pytest.mark.skipif(
    _halops_unsupported_reason() is not None,
    reason=f"HALOps fused top-k: {_halops_unsupported_reason()}",
)


def _config(qkv_dtype: str, num_heads: int) -> fused.FusedTopkConfig:
    config = fused.build_fused_topk_config(
        VLLM_CONFIG,
        num_heads=num_heads,
        head_dim=HEAD_DIM,
        top_k=TOPK,
        use_fp4_kv=qkv_dtype == "mxfp4",
        kv_cache_prefix=PREFIX,
    )
    assert config is not None
    return config


@pytest.fixture(autouse=True)
def fused_topk_enabled(monkeypatch):
    monkeypatch.setenv(fused.ENABLE_ENV_VAR, "1")


@pytest.fixture
def unhooked(monkeypatch):
    """vLLM's own prefill calls; the installed hooks come back afterwards."""
    monkeypatch.setattr(
        sparse_attn_indexer, "fp8_fp4_mqa_logits", deep_gemm.fp8_fp4_mqa_logits
    )
    monkeypatch.setattr(sparse_attn_indexer, "ops", _custom_ops)
    monkeypatch.setattr(flashindexer, "_orig_mqa_logits", None)
    monkeypatch.setattr(fused, "_orig_mqa_logits", None)
    monkeypatch.setattr(fused, "_fused_kernel", None)


@pytest.mark.parametrize(
    "num_heads,num_rows,num_kv,expected",
    [
        (64, 4096, 32768, True),
        (64, 1024, 131072, True),
        # Too few keys per row.
        (64, 8192, 16384, False),
        # Too few rows to fill the GPU.
        (64, 1024, 32768, False),
        (FLASHINDEXER_TOPK_HEADS, 1024, 65536, True),
        (FLASHINDEXER_TOPK_HEADS, 4096, 32768, False),
        (FLASHINDEXER_TOPK_HEADS, 512, 65536, False),
    ],
)
def test_fuses_only_chunks_the_kernel_wins_on(num_heads, num_rows, num_kv, expected):
    assert fused.should_fuse_chunk(num_heads, num_rows, num_kv) is expected


@pytest.mark.parametrize("env", [None, "0"], ids=["unset", "off"])
def test_fused_top_k_is_off_unless_the_env_enables_it(monkeypatch, env):
    if env is None:
        monkeypatch.delenv(fused.ENABLE_ENV_VAR)
    else:
        monkeypatch.setenv(fused.ENABLE_ENV_VAR, env)
    config = fused.build_fused_topk_config(
        VLLM_CONFIG,
        num_heads=64,
        head_dim=HEAD_DIM,
        top_k=TOPK,
        use_fp4_kv=False,
        kv_cache_prefix=PREFIX,
    )
    assert config is None


def test_hooks_pass_other_chunks_through(monkeypatch):
    logits_calls, topk_calls = [], []
    monkeypatch.setattr(
        fused,
        "_orig_mqa_logits",
        lambda *args, **kwargs: logits_calls.append(kwargs) or "logits",
    )
    monkeypatch.setattr(
        fused,
        "_custom_ops",
        SimpleNamespace(top_k_per_row_prefill=lambda *args: topk_calls.append(args)),
    )
    q = (torch.empty(4096, 64, 0), None)
    kv = (torch.empty(32768, 0), None)
    args = (q, kv, "w", "ks", "ke", False)
    # Another model's indexer, or HALOps unavailable.
    assert fused._mqa_logits_hook(*args) == "logits"
    config = fused.FusedTopkConfig(PREFIX, "fp8", 64, HEAD_DIM, TOPK, 148)
    with fused.fused_topk_context(config):
        # A chunk batching several requests.
        monkeypatch.setattr(fused, "_num_reqs_of_chunk", lambda *_: 2)
        assert fused._mqa_logits_hook(*args) == "logits"
        # A chunk too small to win.
        monkeypatch.setattr(fused, "_num_reqs_of_chunk", lambda *_: 1)
        small_q = (torch.empty(512, 64, 0), None)
        assert fused._mqa_logits_hook(small_q, *args[1:]) == "logits"
    assert logits_calls == [{"clean_logits": False}] * 3
    fused._OpsWithFusedTopk.top_k_per_row_prefill("logits", 1, 2, 3, 4, 5, 6, 7)
    assert topk_calls == [("logits", 1, 2, 3, 4, 5, 6, 7)]


def test_hooks_fuse_over_budget_chunks_below_the_win_thresholds(monkeypatch):
    # A low VLLM_SPARSE_INDEXER_MAX_LOGITS_MB lets split_prefill_chunks keep
    # chunks DeepGEMM would materialize over budget.
    monkeypatch.setattr(fused, "_orig_mqa_logits", lambda *a, **k: "logits")
    monkeypatch.setattr(fused, "_num_reqs_of_chunk", lambda *_: 1)
    monkeypatch.setattr(fused, "logits_budget_numel", lambda: 1 << 24)
    monkeypatch.setattr(fused, "_deferred_chunk", None)
    q = (torch.empty(1024, 64, 0), None)
    kv = (torch.empty(32768, 0), None)
    config = fused.FusedTopkConfig(PREFIX, "fp8", 64, HEAD_DIM, TOPK, 148)
    with fused.fused_topk_context(config):
        assert not fused.should_fuse_chunk(64, 1024, 32768)
        logits = fused._mqa_logits_hook(q, kv, torch.empty(1024, 64), 0, 0, False)
    assert logits.shape == (1024, 0)
    assert fused._deferred_chunk.placeholder_logits is logits


def _split(seq_lens, query_lens, request_offset=0):
    return DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks(
        torch.tensor(seq_lens),
        torch.tensor(query_lens),
        1 << 20,
        512 * 1024 * 1024,
        request_offset,
    )


def _planned_split(seq_lens, query_lens, request_offset=0):
    config = fused.FusedTopkConfig(PREFIX, "fp8", 64, HEAD_DIM, TOPK, 148)
    return fused.split_prefill_chunks(
        config,
        DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks,
        torch.tensor(seq_lens),
        torch.tensor(query_lens),
        1 << 20,
        512 * 1024 * 1024,
        request_offset,
    )


def test_split_keeps_fused_requests_whole(monkeypatch):
    monkeypatch.setattr(fused, "max_fused_rows", lambda *_: 12000)
    # vLLM cuts this request into 8 query chunks of 1024 rows.
    assert len(_split([131072], [8192])) == 8
    assert _planned_split([131072], [8192], request_offset=3) == [
        (slice(3, 4), slice(0, 8192))
    ]
    # Rows past the workspace cap start another chunk of the same request.
    assert _planned_split([131072], [20000]) == [
        (slice(0, 1), slice(0, 12000)),
        (slice(0, 1), slice(12000, 20000)),
    ]


def test_split_leaves_unfused_requests_to_vllm(monkeypatch):
    monkeypatch.setattr(fused, "max_fused_rows", lambda *_: 12000)
    short = ([1000, 2000, 3000], [1000, 2000, 3000])
    assert _planned_split(*short, request_offset=2) == _split(*short, 2)
    seq_lens = [1000, 2000, 131072, 3000, 65536, 70000]
    query_lens = [1000, 2000, 8192, 3000, 128, 8192]
    assert _planned_split(seq_lens, query_lens, request_offset=1) == [
        *_split(seq_lens[:2], query_lens[:2], 1),
        (slice(3, 4), slice(0, 8192)),
        # Request 4 has too few rows to fuse and batches with request 3.
        *_split(seq_lens[3:5], query_lens[3:5], 4),
        (slice(6, 7), slice(0, 8192)),
    ]


def test_builder_splits_with_vllm_without_the_fused_top_k(monkeypatch):
    monkeypatch.setattr(fused, "max_fused_rows", lambda *_: 12000)
    builder = object.__new__(Ernie5_5IndexerMetadataBuilder)
    args = (torch.tensor([131072]), torch.tensor([8192]), 1 << 20, 512 << 20)
    builder.split_config = None
    assert builder._split_indexer_prefill_chunks(*args) == _split([131072], [8192])
    builder.split_config = fused.FusedTopkConfig(PREFIX, "fp8", 64, HEAD_DIM, TOPK, 148)
    assert builder._split_indexer_prefill_chunks(*args, request_offset=2) == [
        (slice(2, 3), slice(0, 8192))
    ]


@requires_halops_install
@pytest.mark.parametrize("qkv_dtype", ["fp8", "mxfp4"])
def test_default_budget_keeps_an_8k_token_chunk_whole(qkv_dtype):
    config = fused.FusedTopkConfig(PREFIX, qkv_dtype, 64, HEAD_DIM, TOPK, 148)
    assert fused.max_fused_rows(config, 1 << 20) >= 8192


def test_the_most_heads_config_splits_a_shared_prefill():
    gated = fused.FusedTopkConfig(
        PREFIX, "fp8", FLASHINDEXER_TOPK_HEADS, HEAD_DIM, TOPK, 148
    )
    full = dataclasses.replace(gated, num_heads=64)
    # MTP: the target indexers keep 8 FlashIndexer heads, the MTP indexer 64.
    assert fused.select_split_config([gated, full, gated]) is full
    assert fused.select_split_config([gated]) is gated
    # An unfused config needs vLLM's chunks, as does no config.
    assert fused.select_split_config([gated, None]) is None
    assert fused.select_split_config([]) is None


@requires_halops_gpu
@pytest.mark.parametrize("gated", [False, True], ids=["gate_off", "gate_on"])
def test_loaded_gate_decides_the_fused_heads(monkeypatch, gated):
    warmed_up = []
    monkeypatch.setattr(attention, "warmup_fused_topk", warmed_up.append)
    indexer = SimpleNamespace(
        n_head=64,
        head_dim=HEAD_DIM,
        use_fp4_kv=False,
        linear_gate=nn.Linear(HIDDEN_SIZE, 64) if gated else None,
        indexer_op=SimpleNamespace(topk_tokens=TOPK),
        k_cache=SimpleNamespace(prefix=PREFIX, fused_topk_config=None),
    )
    Ernie5_5Indexer.setup_fused_topk(indexer, VLLM_CONFIG)
    config = indexer.k_cache.fused_topk_config
    assert config.num_heads == (FLASHINDEXER_TOPK_HEADS if gated else 64)
    assert config.kv_cache_prefix == PREFIX
    assert warmed_up == [config]


def test_num_reqs_of_chunk_finds_the_chunk_of_narrowed_bounds(monkeypatch):
    chunks = [
        SimpleNamespace(cu_seqlen_ks=torch.zeros(6, dtype=torch.int32), num_reqs=3),
        SimpleNamespace(cu_seqlen_ks=torch.zeros(4, dtype=torch.int32), num_reqs=1),
    ]
    metadata = {PREFIX: SimpleNamespace(prefill=SimpleNamespace(chunks=chunks))}
    context = SimpleNamespace(attn_metadata=metadata)
    monkeypatch.setattr(fused, "get_forward_context", lambda: context)
    assert fused._num_reqs_of_chunk(PREFIX, chunks[0].cu_seqlen_ks) == 3
    # A TP row shard scores a slice of its chunk's bounds.
    assert fused._num_reqs_of_chunk(PREFIX, chunks[0].cu_seqlen_ks[4:]) == 3
    assert fused._num_reqs_of_chunk(PREFIX, chunks[1].cu_seqlen_ks[1:3]) == 1
    assert fused._num_reqs_of_chunk(PREFIX, torch.zeros(4, dtype=torch.int32)) is None
    assert fused._num_reqs_of_chunk("other.k_cache", chunks[1].cu_seqlen_ks) is None
    # The memory-profiling dummy run carries no metadata dict.
    context.attn_metadata = None
    assert fused._num_reqs_of_chunk(PREFIX, chunks[1].cu_seqlen_ks) is None


@requires_halops_install
@pytest.mark.parametrize("gate_first", [False, True])
def test_fused_top_k_runs_beneath_flashindexer(unhooked, gate_first):
    if gate_first:
        flashindexer.patch_sparse_attn_indexer_flashindexer()
        fused._install_prefill_hooks()
    else:
        fused._install_prefill_hooks()
        flashindexer.patch_sparse_attn_indexer_flashindexer()
    # FlashIndexer selects the heads, then the fused hook scores the selected ones.
    assert (
        sparse_attn_indexer.fp8_fp4_mqa_logits
        is flashindexer._flashindexer_fp8_fp4_mqa_logits
    )
    assert flashindexer._orig_mqa_logits is fused._mqa_logits_hook
    assert fused._orig_mqa_logits is deep_gemm.fp8_fp4_mqa_logits
    # Every other custom op still reaches vLLM's.
    assert isinstance(sparse_attn_indexer.ops, fused._OpsWithFusedTopk)
    assert (
        sparse_attn_indexer.ops.cp_gather_indexer_k_quant_cache
        is _custom_ops.cp_gather_indexer_k_quant_cache
    )


def _chunk(qkv_dtype: str, num_heads: int, num_rows: int, num_kv: int, head: bool):
    """One request's prefill chunk: its first or last `num_rows` queries."""
    kwargs = {"device": "cuda"}
    if qkv_dtype == "mxfp4":
        width, ue8m0 = HEAD_DIM // 2, 0x7F7F7F7F
        q = torch.randint(0, 255, (num_rows, num_heads, width), **kwargs)
        k = torch.randint(0, 255, (num_kv, width), **kwargs)
        q, k = q.to(torch.uint8).view(torch.int8), k.to(torch.uint8).view(torch.int8)
        q_sf = torch.full((num_rows, num_heads), ue8m0, dtype=torch.int32, **kwargs)
        k_sf = torch.full((num_kv,), ue8m0, dtype=torch.int32, **kwargs)
    else:
        q = torch.randn(num_rows, num_heads, HEAD_DIM, **kwargs)
        q = q.to(torch.float8_e4m3fn)
        k = torch.randn(num_kv, HEAD_DIM, **kwargs).to(torch.float8_e4m3fn)
        q_sf, k_sf = None, torch.rand(num_kv, **kwargs) + 0.5
    weights = torch.randn(num_rows, num_heads, **kwargs)
    first = 0 if head else num_kv - num_rows
    positions = torch.arange(first, first + num_rows, **kwargs)
    ks = torch.zeros(num_rows, dtype=torch.int32, **kwargs)
    # The ERNIE metadata builder leaves the sliding window out of the indexer.
    ke = (positions + 1 - WINDOW).clamp_min(0).int()
    return (q, q_sf), (k, k_sf), weights, ks, ke


def _reference_top_k(q, kv, weights, ks, ke) -> torch.Tensor:
    logits = deep_gemm.fp8_fp4_mqa_logits(q, kv, weights, ks, ke, clean_logits=False)
    out = torch.full((ks.numel(), TOPK), -1, dtype=torch.int32, device="cuda")
    _custom_ops.top_k_per_row_prefill(
        logits, ks, ke, out, ks.numel(), logits.stride(0), logits.stride(1), TOPK
    )
    return out


def _hooked_top_k(q, kv, weights, ks, ke) -> torch.Tensor:
    """The vLLM prefill loop's two calls, through whatever hooks are installed."""
    logits = sparse_attn_indexer.fp8_fp4_mqa_logits(
        q, kv, weights, ks, ke, clean_logits=False
    )
    # Deferred to the top-k call instead of materialized.
    assert logits.numel() == 0
    out = torch.full((ks.numel(), TOPK), 7, dtype=torch.int32, device="cuda")
    sparse_attn_indexer.ops.top_k_per_row_prefill(
        logits, ks, ke, out, ks.numel(), logits.stride(0), logits.stride(1), TOPK
    )
    return out


def _assert_same_top_k(out, ref, lens, num_kv):
    short = lens < TOPK
    if short.any():
        # merge_swa_topk reads a short row's keys from its first `len` columns.
        cols = torch.arange(TOPK, device="cuda").expand(int(short.sum()), -1)
        valid = cols < lens[short, None]
        got = torch.where(valid, out[short], TOPK + num_kv).sort(dim=-1).values
        assert torch.equal(got[valid], cols[valid].int())
        assert (out[short][~valid] == -1).all()
    full = ~short
    if full.any():
        assert (out[full] >= 0).all()
        assert (out[full] < lens[full, None]).all()
        picked = torch.zeros(int(full.sum()), num_kv, dtype=torch.bool, device="cuda")
        picked.scatter_(1, ref[full].long(), True)
        # Both kernels score in FP32 but accumulate in different orders, so
        # near-tied keys at the cut can swap.
        assert picked.gather(1, out[full].long()).float().mean() > 0.99


@requires_halops_gpu
@pytest.mark.parametrize("num_heads", [64, FLASHINDEXER_TOPK_HEADS])
@pytest.mark.parametrize("qkv_dtype", ["fp8", "mxfp4"])
def test_warmup_runs_under_the_model_default_dtype(unhooked, qkv_dtype, num_heads):
    config = _config(qkv_dtype, num_heads)
    # vLLM loads, and so warms up, under the model's default dtype.
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        fused.warmup_fused_topk(config)
    finally:
        torch.set_default_dtype(previous)


@requires_halops_gpu
@pytest.mark.parametrize("head", [False, True], ids=["tail_rows", "head_rows"])
@pytest.mark.parametrize("gated", [False, True], ids=["gate_off", "gate_on"])
@pytest.mark.parametrize("qkv_dtype", ["fp8", "mxfp4"])
def test_fused_top_k_matches_deepgemm_logits_and_top_k(
    unhooked, monkeypatch, qkv_dtype, gated, head
):
    torch.manual_seed(0)
    config = _config(qkv_dtype, FLASHINDEXER_TOPK_HEADS if gated else 64)
    flashindexer.patch_sparse_attn_indexer_flashindexer()
    # The forward-context lookup needs real metadata; this is one request.
    monkeypatch.setattr(fused, "_num_reqs_of_chunk", lambda *_: 1)
    num_kv = fused.min_fused_kv_tokens(config.num_heads)
    num_rows = fused.MIN_ROWS_TIMES_KV_TOKENS // num_kv
    q, kv, weights, ks, ke = _chunk(qkv_dtype, 64, num_rows, num_kv, head)
    gate = contextlib.nullcontext()
    if gated:
        hidden = torch.randn(num_rows, HIDDEN_SIZE, device="cuda").bfloat16()
        indexer = SimpleNamespace(
            linear_gate=SimpleNamespace(
                weight=torch.randn(64, HIDDEN_SIZE, device="cuda").bfloat16()
                / HIDDEN_SIZE**0.5
            ),
            flashindexer_bias=torch.randn(64, device="cuda"),
        )
        gate = flashindexer_context(indexer, hidden, weights)
        q_sel, q_sf_sel, w_sel = flashindexer_select_heads_fused(
            q[0],
            weights,
            indexer.linear_gate.weight,
            indexer.flashindexer_bias,
            hidden,
            q[1],
        )
        ref = _reference_top_k((q_sel, q_sf_sel), kv, w_sel, ks, ke)
    else:
        ref = _reference_top_k(q, kv, weights, ks, ke)

    with fused.fused_topk_context(config), gate:
        out = _hooked_top_k(q, kv, weights, ks, ke)
    _assert_same_top_k(out, ref, (ke - ks).long(), num_kv)
