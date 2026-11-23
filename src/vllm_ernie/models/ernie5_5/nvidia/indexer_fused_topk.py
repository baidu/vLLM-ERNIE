# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HALOps fused MQA logits + top-k for large ERNIE 5.5 DSA indexer prefill chunks."""

from __future__ import annotations

import contextlib
import dataclasses
import os
import time
from collections.abc import Callable, Iterator

import torch

import vllm.envs as envs
from vllm import _custom_ops
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform

from . import flashindexer

logger = init_logger(__name__)

# "1" enables the fused kernel; the default "0" keeps DeepGEMM logits +
# top_k_per_row_prefill for every prefill chunk.
ENABLE_ENV_VAR = "VLLM_ERNIE_HALOPS_INDEXER_TOPK"

# Measured on SM100 with vLLM-shaped single-request chunks, FP8 and MXFP4: the
# fused kernel is 1.05-1.5x faster above these sizes and up to 5x slower below.
MIN_KV_TOKENS_64_HEADS = 32768
MIN_KV_TOKENS_FEW_HEADS = 65536
MIN_ROWS_TIMES_KV_TOKENS = 1 << 26

# A prefill chunk as vLLM's indexer chunker returns it: (request slice, query-row
# slice within the request).
PrefillChunkSlices = tuple[slice, slice]


@dataclasses.dataclass(frozen=True)
class FusedTopkConfig:
    """The fused top-k settings of one indexer."""

    kv_cache_prefix: str
    qkv_dtype: str
    # Heads the logits kernel sees: 8 behind a FlashIndexer gate, else all.
    num_heads: int
    head_dim: int
    top_k: int
    num_sms: int


@dataclasses.dataclass(frozen=True)
class _DeferredChunk:
    """A chunk's logits inputs, held until its top-k call."""

    placeholder_logits: torch.Tensor
    q: torch.Tensor
    q_scale: torch.Tensor | None
    kv: tuple[torch.Tensor, torch.Tensor]
    weights: torch.Tensor
    num_sms: int


_active_config: FusedTopkConfig | None = None
_deferred_chunk: _DeferredChunk | None = None
_orig_mqa_logits = None
_fused_kernel = None
_warmed_up_keys: set[tuple] = set()


def min_fused_kv_tokens(num_heads: int) -> int:
    return MIN_KV_TOKENS_64_HEADS if num_heads >= 64 else MIN_KV_TOKENS_FEW_HEADS


def should_fuse_chunk(num_heads: int, num_rows: int, num_kv_tokens: int) -> bool:
    """Whether a single-request chunk is large enough for the fused kernel."""
    return (
        num_kv_tokens >= min_fused_kv_tokens(num_heads)
        and num_rows * num_kv_tokens >= MIN_ROWS_TIMES_KV_TOKENS
    )


def logits_budget_numel() -> int:
    """FP32 logits elements vLLM's prefill chunker and memory profile allow."""
    return envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024 // 4


def max_fused_rows(config: FusedTopkConfig, num_kv_tokens: int) -> int:
    """Rows per fused chunk whose workspace fits the profiled logits budget."""
    from halops.indexer.kernels.mqa_logits_fused_topk.config import (
        bounded_workspace_bytes,
    )

    workspace_bytes_per_row = bounded_workspace_bytes(
        1, num_kv_tokens, config.top_k, qkv_dtype=config.qkv_dtype
    )
    return max(1, logits_budget_numel() * 4 // workspace_bytes_per_row)


def select_split_config(
    configs: list[FusedTopkConfig | None],
) -> FusedTopkConfig | None:
    """The config whose thresholds split the prefill shared by `configs`.

    One metadata builder serves layers with different head counts (MTP: an
    8-head target, a 64-head draft). More heads fuse shorter requests, so the
    most-heads config keeps whole every request some layer fuses; a layer with
    fewer heads still fuses such a request, as an over-budget chunk. None if
    any layer is unfused: that layer would materialize a whole request's logits.
    """
    if not configs or None in configs:
        return None
    return max(configs, key=lambda config: config.num_heads)


def split_prefill_chunks(
    config: FusedTopkConfig,
    vllm_split: Callable[..., list[PrefillChunkSlices]],
    seq_lens_cpu: torch.Tensor,
    query_lens_cpu: torch.Tensor,
    workspace_size: int,
    max_logits_bytes: int,
    request_offset: int = 0,
) -> list[PrefillChunkSlices]:
    """vLLM's prefill chunking, with each fused request kept in one chunk.

    `vllm_split` chunks every other request. A fused request only splits where
    its workspace would outgrow the logits budget, which at the default 512 MB
    takes about 12K query tokens.
    """
    chunks: list[PrefillChunkSlices] = []
    first_unplanned_req = 0
    for req, (num_rows, num_kv_tokens) in enumerate(
        zip(query_lens_cpu.tolist(), seq_lens_cpu.tolist(), strict=True)
    ):
        if not should_fuse_chunk(config.num_heads, num_rows, num_kv_tokens):
            continue
        if first_unplanned_req < req:
            chunks += vllm_split(
                seq_lens_cpu[first_unplanned_req:req],
                query_lens_cpu[first_unplanned_req:req],
                workspace_size,
                max_logits_bytes,
                request_offset + first_unplanned_req,
            )
        req_slice = slice(request_offset + req, request_offset + req + 1)
        rows_per_chunk = max_fused_rows(config, num_kv_tokens)
        chunks += [
            (req_slice, slice(row, min(row + rows_per_chunk, num_rows)))
            for row in range(0, num_rows, rows_per_chunk)
        ]
        first_unplanned_req = req + 1
    if first_unplanned_req < len(seq_lens_cpu):
        chunks += vllm_split(
            seq_lens_cpu[first_unplanned_req:],
            query_lens_cpu[first_unplanned_req:],
            workspace_size,
            max_logits_bytes,
            request_offset + first_unplanned_req,
        )
    return chunks


def build_fused_topk_config(
    vllm_config: VllmConfig,
    *,
    num_heads: int,
    head_dim: int,
    top_k: int,
    use_fp4_kv: bool,
    kv_cache_prefix: str,
) -> FusedTopkConfig | None:
    """Config for an indexer the fused kernel can serve, else None.

    Installs the prefill hooks the first time it returns a config.
    """
    if os.getenv(ENABLE_ENV_VAR, "0") != "1" or not current_platform.is_cuda():
        return None
    parallel_config = vllm_config.parallel_config
    if (
        parallel_config.decode_context_parallel_size != 1
        or parallel_config.prefill_context_parallel_size != 1
    ):
        # CP merges each rank's top-k from logits the fused kernel never writes.
        return None
    try:
        from halops.indexer.kernels.mqa_logits_fused_topk.config import (
            unsupported_reason,
        )
    except ImportError:
        logger.info_once("HALOps is not installed; using the unfused indexer top-k.")
        return None
    qkv_dtype = "mxfp4" if use_fp4_kv else "fp8"
    reason = unsupported_reason(num_heads, head_dim, top_k, qkv_dtype=qkv_dtype)
    if reason is not None:
        logger.info_once("HALOps fused indexer top-k is unavailable: %s", reason)
        return None
    _install_prefill_hooks()
    logger.info_once(
        "Using HALOps fused MQA logits + top-k for large %s prefill chunks (%d heads).",
        qkv_dtype,
        num_heads,
    )
    device = torch.cuda.current_device()
    return FusedTopkConfig(
        kv_cache_prefix=kv_cache_prefix,
        qkv_dtype=qkv_dtype,
        num_heads=num_heads,
        head_dim=head_dim,
        top_k=top_k,
        num_sms=torch.cuda.get_device_properties(device).multi_processor_count,
    )


@contextlib.contextmanager
def fused_topk_context(config: FusedTopkConfig | None) -> Iterator[None]:
    """Let the prefill hooks fuse this indexer's chunks."""
    global _active_config
    if config is None:
        yield
        return
    previous_config = _active_config
    _active_config = config
    try:
        yield
    finally:
        _active_config = previous_config


def _num_reqs_of_chunk(kv_cache_prefix: str, row_starts: torch.Tensor) -> int | None:
    """Request count of the prefill chunk these (maybe narrowed) row starts are of."""
    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict) or kv_cache_prefix not in attn_metadata:
        return None
    prefill = attn_metadata[kv_cache_prefix].prefill
    if prefill is None:
        return None
    # vLLM passes a chunk's own `cu_seqlen_ks`, or a TP row shard's slice of it.
    # Every chunk allocates that tensor, so its storage identifies the chunk.
    storage_ptr = row_starts.untyped_storage().data_ptr()
    for chunk in prefill.chunks:
        if chunk.cu_seqlen_ks.untyped_storage().data_ptr() == storage_ptr:
            return chunk.num_reqs
    return None


def _mqa_logits_hook(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits):
    """`fp8_fp4_mqa_logits` that defers large chunks to the fused top-k."""
    global _deferred_chunk
    config = _active_config
    q_values, q_scale = q
    num_rows, num_heads = q_values.shape[:2]
    num_kv_tokens = kv[0].shape[0]
    if (
        config is None
        or clean_logits
        # Only split_prefill_chunks leaves a chunk over the logits budget.
        or not (
            should_fuse_chunk(num_heads, num_rows, num_kv_tokens)
            or num_rows * num_kv_tokens > logits_budget_numel()
        )
        # Batched short requests leave too little work per row to win.
        or _num_reqs_of_chunk(config.kv_cache_prefix, cu_seqlen_ks) != 1
    ):
        return _orig_mqa_logits(
            q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=clean_logits
        )
    # Proof that serving reaches the fused kernel, not just that it is set up.
    logger.info_once("HALOps fused top-k is scoring prefill chunks.")
    placeholder_logits = weights.new_empty((num_rows, 0))
    _deferred_chunk = _DeferredChunk(
        placeholder_logits, q_values, q_scale, kv, weights, config.num_sms
    )
    return placeholder_logits


class _OpsWithFusedTopk:
    """`vllm._custom_ops`, with the prefill top-k finishing deferred chunks."""

    def __getattr__(self, name: str):
        return getattr(_custom_ops, name)

    @staticmethod
    def top_k_per_row_prefill(
        logits: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        raw_topk_indices: torch.Tensor,
        num_rows: int,
        stride0: int,
        stride1: int,
        topk_tokens: int,
    ) -> None:
        global _deferred_chunk
        chunk = _deferred_chunk
        if chunk is None or logits is not chunk.placeholder_logits:
            _custom_ops.top_k_per_row_prefill(
                logits,
                cu_seqlen_ks,
                cu_seqlen_ke,
                raw_topk_indices,
                num_rows,
                stride0,
                stride1,
                topk_tokens,
            )
            return
        _deferred_chunk = None
        # Row-relative indices with -1 padding, as top_k_per_row_prefill writes.
        # A row shorter than top_k holds its keys, unordered, in the first columns.
        _fused_kernel(
            chunk.q,
            chunk.q_scale,
            chunk.kv,
            chunk.weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            None,
            topk_tokens,
            scores_bf16=False,
            num_sms=chunk.num_sms,
            topk_indices=raw_topk_indices,
        )


def _install_prefill_hooks() -> None:
    """Hook the prefill logits and top-k calls of vLLM's sparse indexer."""
    global _orig_mqa_logits, _fused_kernel
    if _fused_kernel is not None:
        return
    from halops.indexer import mqa_logits_fused_topk

    from vllm.model_executor.layers import sparse_attn_indexer

    current = sparse_attn_indexer.fp8_fp4_mqa_logits
    if getattr(current, "_ernie5_flashindexer", False):
        # FlashIndexer narrows the heads first, so the fused top-k goes beneath it.
        _orig_mqa_logits = flashindexer._orig_mqa_logits
        flashindexer._orig_mqa_logits = _mqa_logits_hook
    else:
        _orig_mqa_logits = current
        sparse_attn_indexer.fp8_fp4_mqa_logits = _mqa_logits_hook
    sparse_attn_indexer.ops = _OpsWithFusedTopk()
    _fused_kernel = mqa_logits_fused_topk


def warmup_fused_topk(config: FusedTopkConfig) -> None:
    """Compile the fused top-k kernels `config` serves, once per process."""
    device = torch.cuda.current_device()
    key = (config.qkv_dtype, config.num_heads, config.top_k, config.num_sms, device)
    if key in _warmed_up_keys:
        return
    from halops.indexer.kernels.mqa_logits_fused_topk.config import (
        warmup_row_multiples,
    )

    start_time = time.perf_counter()
    # halops.indexer.warmup also builds the output-offset variants, which the
    # indexer never passes and which nearly triple the compile time.
    row_multiples = warmup_row_multiples(config.num_heads)
    num_rows = max(row_multiples) * config.num_sms
    num_kv_tokens = min_fused_kv_tokens(config.num_heads)
    num_heads = config.num_heads
    use_fp4 = config.qkv_dtype == "mxfp4"
    # Explicit FP32: model loading sets the default dtype to the model's.
    fp32 = dict(dtype=torch.float32, device=device)
    packed_dim = config.head_dim // 2 if use_fp4 else config.head_dim
    qkv_dtype = torch.int8 if use_fp4 else torch.float8_e4m3fn
    q = torch.zeros((num_rows, num_heads, packed_dim), dtype=qkv_dtype, device=device)
    k = torch.zeros((num_kv_tokens, packed_dim), dtype=qkv_dtype, device=device)
    if use_fp4:
        # UE8M0 scale 1.0 in each of the four packed bytes.
        unit_scales = dict(fill_value=0x7F7F7F7F, dtype=torch.int32, device=device)
        q_scale = torch.full((num_rows, num_heads), **unit_scales)
        k_scale = torch.full((num_kv_tokens,), **unit_scales)
    else:
        q_scale = None
        k_scale = torch.ones(num_kv_tokens, **fp32)
    weights = torch.ones((num_rows, num_heads), **fp32)
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device=device)
    row_ends = torch.full((num_rows,), num_kv_tokens, dtype=torch.int32, device=device)
    topk_indices = torch.empty(
        (num_rows, config.top_k), dtype=torch.int32, device=device
    )
    # Row counts on every tile choice; any other count reuses these kernels.
    for multiple in row_multiples:
        rows = multiple * config.num_sms
        _fused_kernel(
            q[:rows],
            None if q_scale is None else q_scale[:rows],
            (k, k_scale),
            weights[:rows],
            row_starts[:rows],
            row_ends[:rows],
            None,
            config.top_k,
            scores_bf16=False,
            num_sms=config.num_sms,
            topk_indices=topk_indices[:rows],
        )
    torch.cuda.synchronize(device)
    _warmed_up_keys.add(key)
    logger.info(
        "Warmed up HALOps fused indexer top-k (%s, %d heads) in %.1f s.",
        config.qkv_dtype,
        num_heads,
        time.perf_counter() - start_time,
    )
