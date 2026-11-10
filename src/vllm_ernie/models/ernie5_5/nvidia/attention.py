# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HCA and DSA attention for ERNIE 5 (NVIDIA)."""

import copy
import dataclasses
import math
from typing import cast

import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import downcast_to_mxfp
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.common.ops.fused_indexer_q import MXFP4_BLOCK_SIZE
from vllm.models.deepseek_v4.compressor import DeepseekCompressor
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferMLAAttention,
)
from vllm.models.deepseek_v4.nvidia.flashmla import DeepseekV4FlashMLAAttention
from vllm.models.deepseek_v4.nvidia.ops.o_proj import deep_gemm_fp8_o_proj
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata, has_deep_gemm
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseImpl,
    FlashMLASparseMetadata,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerMetadataBuilder,
    dsa_indexer_uses_fp4,
    get_max_prefill_buffer_size,
)
from vllm.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
)
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.workspace import current_workspace_manager

from vllm_ernie.models.ernie5_5.config import Ernie5_5Config

from .quant_config import keeps_fp8_wo_a

logger = init_logger(__name__)

VHA_RANK = 16
# FlashMLA sparse and the paged MQA logits kernels both use 64-token pages.
DSA_BLOCK_SIZE = 64
INDEXER_QUANT_BLOCK = 128


def hadamard_rows(x: torch.Tensor) -> torch.Tensor:
    """Hadamard transform per row, zero-padded around a hadacore tail overrun."""
    num_rows = x.shape[0]
    pad = -num_rows % max(1, 512 // x.shape[1])
    if pad:
        x = nn.functional.pad(x, (0, 0, 0, pad))
    return ops.hadacore_transform(x, inplace=False)[:num_rows]


def _check_attn_tp(config: Ernie5_5Config) -> None:
    tp_size = get_tensor_model_parallel_world_size()
    for name in ("o_groups", "num_attention_heads", "hybrid_mla_num_attention_heads"):
        if getattr(config, name) % tp_size:
            raise ValueError(
                f"ERNIE 5.5 attention TP size {tp_size} must divide "
                f"{name}={getattr(config, name)}."
            )


class VHAPostMix(nn.Module):
    """Head mixing ``o[h] <- sum_g m[h, g] * o[g]`` with ``m = I + V @ U^T``."""

    def __init__(self, num_heads: int, rank: int = VHA_RANK) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        num_local_heads = num_heads // self.tp_size
        self.U = nn.Parameter(torch.zeros(num_local_heads, rank), requires_grad=False)
        self.V = nn.Parameter(torch.zeros(num_local_heads, rank), requires_grad=False)
        for param in (self.U, self.V):
            set_weight_attrs(param, {"weight_loader": self._load_local_heads})
        if self.tp_size == 1:
            self.register_buffer("m", torch.eye(num_heads), persistent=False)

    def _load_local_heads(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        rows = param.shape[0]
        param.data.copy_(loaded[self.tp_rank * rows : (self.tp_rank + 1) * rows])

    def fold(self) -> None:
        if self.tp_size != 1:
            return
        # Recomputed from U/V each time so repeated calls and weight
        # reloads stay idempotent; in place to keep CUDA graph addresses.
        eye = torch.eye(self.U.shape[0], dtype=self.U.dtype, device=self.U.device)
        self.m.copy_(torch.addmm(eye, self.V, self.U.t()))

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        # o: [num_tokens, num_local_heads, head_dim]
        if self.tp_size == 1:
            return torch.matmul(self.m, o)
        # TP: o + V @ all_reduce(U^T @ o) over this rank's heads, in fp32.
        num_tokens = o.shape[0]
        u = self.U.t().expand(num_tokens, -1, -1)
        z = tensor_model_parallel_all_reduce(torch.bmm(u, o, out_dtype=torch.float32))
        v = self.V.float().expand(num_tokens, -1, -1)
        return torch.bmm(v, z).add_(o).to(o.dtype)


def fold_vha(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, VHAPostMix):
            module.fold()


class Ernie5_5Compressor(DeepseekCompressor):
    """Keeps C128 compression in piecewise CUDA graphs (upstream #59709)."""

    def forward(self, *args, **kwargs):
        metadata = get_forward_context().attn_metadata
        if torch.cuda.is_current_stream_capturing() and isinstance(metadata, dict):
            name = self.state_cache.prefix
            state = metadata.get(name)
            if state is not None and state.c128_boundary is False:
                metadata[name] = dataclasses.replace(state, c128_boundary=None)
        return super().forward(*args, **kwargs)


class Ernie5_5HCAAttentionMixin:
    """Adds the ERNIE 5 gate and VHA to an HCA attention class."""

    def __init__(self, vllm_config: VllmConfig, prefix: str, **kwargs) -> None:
        _check_attn_tp(vllm_config.model_config.hf_config)
        super().__init__(vllm_config, prefix, **kwargs)  # type: ignore[call-arg]
        # wo_a stays FP8 only for 128x128 block-FP8 checkpoints.
        if not keeps_fp8_wo_a(vllm_config.quant_config):
            self.wo_a = ColumnParallelLinear(
                self.wo_a.input_size,
                self.wo_a.output_size,
                bias=False,
                quant_config=None,
                return_bias=False,
                prefix=f"{prefix}.wo_a",
            )
            self.wo_a.is_bmm = True
            self.wo_a.bmm_batch_size = self.n_local_groups
        # Split by group like wo_a.
        self.o_gate = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=vllm_config.quant_config,
            return_bias=False,
            prefix=f"{prefix}.o_gate",
        )
        self.vha = VHAPostMix(self.n_heads)
        if getattr(self, "compressor", None) is not None:
            self.compressor.__class__ = Ernie5_5Compressor

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        qr_kv, kv_score, indexer_kv_score, indexer_weights = (
            self._run_parallel_input_projections(hidden_states)
        )
        qr, qr_scale, kv = self._split_qkv_and_norm(qr_kv)
        gate = self.o_gate(qr)
        self._prepare_and_attn_fn(
            hidden_states,
            qr,
            kv,
            qr_scale,
            kv_score,
            indexer_kv_score,
            indexer_weights,
            positions,
            o_padded,
        )
        # VHA acts on the head axis and inverse RoPE on the dim axis, so
        # mixing before the fused inverse-RoPE o_proj is exact.
        o = self.vha(o_padded[:, : self.n_local_heads, :])
        return deep_gemm_fp8_o_proj(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.wo_a,
            lambda z: self.wo_b(z * torch.sigmoid(gate)),
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            o_lora_rank=self.o_lora_rank,
            einsum_recipe=self._einsum_recipe,
            tma_aligned_scales=self._tma_aligned_scales,
        )


class Ernie5_5HCAFlashInferAttention(
    Ernie5_5HCAAttentionMixin, DeepseekV4FlashInferMLAAttention
):
    """HCA layer on FlashInfer TRTLLM-gen sparse MLA (bf16 KV cache)."""


class Ernie5_5HCAFlashMLAAttention(
    Ernie5_5HCAAttentionMixin, DeepseekV4FlashMLAAttention
):
    """HCA layer on FlashMLA sparse MLA (``fp8_ds_mla`` KV cache)."""


def select_ernie5_5_hca_cls(kv_cache_dtype: str) -> type[DeepseekV4Attention]:
    """FlashInfer for the plain bf16 KV row, FlashMLA for ``fp8_ds_mla``."""
    if kv_cache_dtype.startswith("fp8"):
        return Ernie5_5HCAFlashMLAAttention
    return Ernie5_5HCAFlashInferAttention


def packed_block_stride_alignment(vllm_config: VllmConfig) -> int:
    """Byte alignment of the block stride that every layer's page shares.

    The paged MQA logits kernel needs 512B, and the FlashInfer HCA and
    FlashMLA sparse kernels view their bf16 KV caches as flat rows, so the
    stride must also be a whole number of HCA and DSA rows.
    """
    hf_config = vllm_config.model_config.hf_config
    hca_row = hf_config.head_dim * 2
    dsa_row = (hf_config.mla_kv_lora_rank + hf_config.mla_qk_rope_head_dim) * 2
    return math.lcm(512, hca_row, dsa_row)


def _dsa_page_spec(spec: KVCacheSpec, vllm_config: VllmConfig) -> KVCacheSpec:
    return dataclasses.replace(
        spec,
        block_size=DSA_BLOCK_SIZE,
        block_stride_alignment=packed_block_stride_alignment(vllm_config),
    )


@triton.jit
def _merge_swa_topk_kernel(
    topk_ptr,
    out_ptr,
    positions_ptr,
    topk_stride,
    out_stride,
    WINDOW: tl.constexpr,
    TOPK: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    seq_len = tl.load(positions_ptr + row).to(tl.int32) + 1
    indexer_len = tl.minimum(tl.maximum(seq_len - WINDOW, 0), TOPK)
    swa_len = tl.minimum(seq_len, WINDOW)
    cols = tl.arange(0, BLOCK)
    is_topk = cols < indexer_len
    topk = tl.load(topk_ptr + row * topk_stride + cols, mask=is_topk, other=-1)
    swa_idx = seq_len - swa_len + cols - indexer_len
    vals = tl.where(is_topk, topk, tl.where(cols < indexer_len + swa_len, swa_idx, -1))
    tl.store(out_ptr + row * out_stride + cols, vals, mask=cols < WIDTH)


def merge_swa_topk(
    topk_indices: torch.Tensor,
    out: torch.Tensor,
    positions: torch.Tensor,
    window: int,
) -> None:
    """Write each indexer top-k row followed by the sliding window into `out`.

    Each row becomes ``[topk[:n] | seq-w .. seq-1 | -1 ...]`` with request-local
    token indices, where ``n = min(max(seq - window, 0), topk)``, ``w =
    min(seq, window)`` and ``seq = position + 1``. The indexer writes its
    output assuming a row stride of ``topk``, so it cannot fill `out` directly.

    Args:
        topk_indices: ``[num_tokens, topk]`` int32 indexer output.
        out: ``[num_tokens, width]`` int32 buffer, ``width >= topk + window``.
        positions: ``[num_tokens]`` token positions.
        window: Sliding window size.

    """
    num_tokens = positions.shape[0]
    if num_tokens == 0:
        return
    topk, width = topk_indices.shape[1], out.shape[1]
    assert width >= topk + window
    _merge_swa_topk_kernel[(num_tokens,)](
        topk_indices,
        out,
        positions,
        topk_indices.stride(0),
        out.stride(0),
        WINDOW=window,
        TOPK=topk,
        WIDTH=width,
        BLOCK=triton.next_power_of_2(width),
    )


class Ernie5_5IndexerMetadataBuilder(DeepseekV32IndexerMetadataBuilder):
    """Excludes the trailing sliding window from the indexer KV range."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.window = self.vllm_config.model_config.hf_config.sliding_window
        self.windowed_seq_lens_buffer = torch.zeros_like(self.decode_seq_lens_buffer)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV32IndexerMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        if metadata.prefill is not None:
            metadata.prefill.chunks = [
                dataclasses.replace(
                    chunk,
                    cu_seqlen_ke=torch.maximum(
                        chunk.cu_seqlen_ke - self.window, chunk.cu_seqlen_ks
                    ),
                )
                for chunk in metadata.prefill.chunks
            ]
        if metadata.decode is not None:
            seq_lens = metadata.decode.seq_lens
            windowed = self.windowed_seq_lens_buffer[: seq_lens.numel()].view(
                seq_lens.shape
            )
            torch.sub(seq_lens, self.window, out=windowed)
            windowed.clamp_min_(0)
            schedule_metadata = metadata.decode.schedule_metadata
            if current_platform.is_cuda() and has_deep_gemm():
                schedule = get_paged_mqa_logits_metadata(
                    windowed,
                    self.kv_cache_spec.num_states,
                    self.num_sms,
                    indices=metadata.decode.indices,
                )
                schedule_metadata = self.scheduler_metadata_buffer[: schedule.shape[0]]
                schedule_metadata[:] = schedule
            metadata.decode = dataclasses.replace(
                metadata.decode,
                seq_lens=windowed,
                schedule_metadata=schedule_metadata,
            )
        return metadata


def _dsa_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
    # DSA layers own a 64-token KV group, so their kernel block stays 64. A
    # multiple (instead of exactly 64) lets KV connectors, which pick one kernel
    # block size for all backends, keep the 256-token manager block.
    return [MultipleOf(DSA_BLOCK_SIZE)]


class Ernie5_5FlashMLASparseImpl(FlashMLASparseImpl):
    """FlashMLA sparse impl that passes the per-head sink to the kernels.

    The kernels fold the sink into the softmax denominator. The base impl
    drops the ``sinks`` kwarg, so both kernel wrappers are overridden.
    """

    def __init__(self, *args, **kwargs) -> None:
        # The base impl only takes explicit keyword arguments.
        self.sinks: torch.Tensor = kwargs.pop("sinks")
        super().__init__(*args, **kwargs)

    def _padded_sinks(self, kernel_heads: int) -> torch.Tensor:
        num_heads = self.sinks.shape[0]
        if kernel_heads == num_heads:
            return self.sinks
        # Built per call so it follows weights loaded after construction;
        # padded lanes get -inf, a no-op sink.
        padded = self.sinks.new_full((kernel_heads,), float("-inf"))
        padded[:num_heads] = self.sinks
        return padded

    def _fp8_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        kernel_metadata: FlashMLASparseMetadata.FP8KernelMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q: [batch, seq_len, num_heads, head_dim]; kernel needs 64 or 128 heads.
        num_heads = q.size(2)
        padded_heads = self.fp8_decode_padded_heads
        if num_heads < padded_heads:
            q_padded = q.new_zeros((q.size(0), q.size(1), padded_heads, q.size(3)))
            q_padded[:, :, :num_heads] = q
            q = q_padded
        out, lse = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2),
            block_table=kernel_metadata.dummy_block_table,
            head_dim_v=512,
            cache_seqlens=kernel_metadata.cache_lens,
            tile_scheduler_metadata=kernel_metadata.scheduler_metadata,
            is_fp8_kvcache=True,
            indices=topk_indices,
            softmax_scale=self.softmax_scale,
            attn_sink=self._padded_sinks(padded_heads),
        )
        return out[:, :, :num_heads], lse[:, :num_heads]

    def _bf16_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_length: torch.Tensor | None = None,
        actual_num_heads: int | None = None,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )
        if actual_num_heads is None:
            actual_num_heads = q.shape[1]
        padded_heads = round_up(actual_num_heads, self.prefill_padding)
        if q.shape[1] < padded_heads:
            # Zero the padded lanes: the kernel reduces across the head group,
            # so uninitialized memory could leak NaNs into the real heads.
            if out is None:
                q_padded = q.new_zeros((num_tokens, padded_heads, q.shape[2]))
            else:
                *_, q_padded, _, _ = current_workspace_manager().get_simultaneous(
                    *self.workspace_specs
                )
                q_padded = q_padded[:num_tokens]
                q_padded[:, actual_num_heads:].zero_()
            q_padded[:, :actual_num_heads] = q
            q = q_padded
        output, _, lse = flash_mla_sparse_fwd(
            q,
            kv_c_and_k_pe_cache,
            topk_indices.view(num_tokens, 1, -1),
            self.softmax_scale,
            attn_sink=self._padded_sinks(padded_heads),
            topk_length=topk_length,
            out=out,
        )
        return output[:, :actual_num_heads], lse[:, :actual_num_heads]


class Ernie5_5FlashMLASparseBackend(FlashMLASparseBackend):
    # Keeps the parent name: shared code keys the fp8_ds_mla KV layout on
    # "FLASHMLA_SPARSE".

    get_supported_kernel_block_sizes = staticmethod(_dsa_kernel_block_sizes)

    @staticmethod
    def get_impl_cls() -> type[Ernie5_5FlashMLASparseImpl]:
        return Ernie5_5FlashMLASparseImpl

    @classmethod
    def supports_sink(cls) -> bool:
        return True


class Ernie5_5IndexerBackend(DeepseekV32IndexerBackend):
    # Keeps the parent name: shared MLA code keys on it.

    get_supported_kernel_block_sizes = staticmethod(_dsa_kernel_block_sizes)

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # The 64-token DSA pages are packed beside the 256-token HCA pages
        # inside each block, so the block dim must be outermost.
        return (KVCacheLayout.BLNHC, KVCacheLayout.BLHNC)

    @staticmethod
    def get_builder_cls() -> type[Ernie5_5IndexerMetadataBuilder]:
        return Ernie5_5IndexerMetadataBuilder


class Ernie5_5IndexerCache(DeepseekV32IndexerCache):
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return _dsa_page_spec(super().get_kv_cache_spec(vllm_config), vllm_config)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Ernie5_5IndexerBackend


@triton.jit
def _store_indexer_fp4(
    values_ptr,
    scales_ptr,
    cache_ptr,
    slots_ptr,
    values_stride,
    scales_stride,
    cache_stride,
    PAGE_SIZE: tl.constexpr,
    VALUE_BYTES: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(slots_ptr + row).to(tl.int64)
    if slot < 0:
        return
    page_base = slot // PAGE_SIZE * cache_stride
    offset = slot % PAGE_SIZE
    values = tl.arange(0, VALUE_BYTES)
    scales = tl.arange(0, SCALE_BYTES)
    # DeepGEMM pages hold all packed values first, then all UE8M0 scales.
    tl.store(
        cache_ptr + page_base + offset * VALUE_BYTES + values,
        tl.load(values_ptr + row * values_stride + values),
    )
    tl.store(
        cache_ptr + page_base + PAGE_SIZE * VALUE_BYTES + offset * SCALE_BYTES + scales,
        tl.load(scales_ptr + row * scales_stride + scales),
    )


class Ernie5_5Indexer(nn.Module):
    """Lightning indexer scoring the tokens older than the sliding window."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Ernie5_5Config,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig,
        topk_indices_buffer: torch.Tensor | None,
        swa_topk_buffer: torch.Tensor | None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.mla_qk_rope_head_dim
        self.use_fp4_kv = dsa_indexer_uses_fp4(vllm_config)
        if self.use_fp4_kv:
            parallel_config = vllm_config.parallel_config
            pcp_size = parallel_config.prefill_context_parallel_size
            dcp_size = parallel_config.decode_context_parallel_size
            # The external FP4 cache writer does not handle CP-sharded inputs.
            if pcp_size != 1 or dcp_size != 1:
                raise NotImplementedError(
                    "ERNIE 5.5 MXFP4 indexer currently requires PCP=DCP=1 "
                    "(prefill_context_parallel_size=1 and "
                    "decode_context_parallel_size=1). "
                    f"Got PCP={pcp_size}, DCP={dcp_size}."
                )
        logger.info_once(
            "Using %s indexer cache for Lightning Indexer.",
            "MXFP4" if self.use_fp4_kv else "FP8",
        )
        self.wq_b = ReplicatedLinear(
            config.mla_q_lora_rank,
            self.n_head * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.wk_weights_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [self.head_dim, self.n_head],
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.wk_weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.rotary_emb = get_rope(
            self.rope_dim,
            max_position=config.max_position_embeddings,
            is_neox_style=True,
            rope_parameters=config.mla_rope_parameters,
        )
        # The hadacore transform is orthonormal, so only the usual softmax and
        # head-count scaling is folded into the weights.
        self.weights_scale = self.head_dim**-0.5 * self.n_head**-0.5
        # Read by the sparse MLA impl through its indexer, so this holds the
        # merged rows; the op writes its top-k to a buffer of row stride topk.
        self.topk_indices_buffer = swa_topk_buffer
        if self.use_fp4_kv:
            # MXFP4 stores two values per byte plus one UE8M0 byte per 32 values.
            # head_dim bytes = 64 packed values + 4 UE8M0 scales = 68.
            k_cache_head_dim = self.head_dim // 2 + self.head_dim // MXFP4_BLOCK_SIZE
        else:
            # head_dim bytes = 128 fp8 + 4 fp32 scale = 132.
            k_cache_head_dim = self.head_dim + self.head_dim // INDEXER_QUANT_BLOCK * 4
        self.k_cache = Ernie5_5IndexerCache(
            head_dim=k_cache_head_dim,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )
        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            INDEXER_QUANT_BLOCK,
            "ue8m0",
            config.index_topk,
            self.head_dim,
            vllm_config.model_config.max_model_len,
            get_max_prefill_buffer_size(vllm_config),
            topk_indices_buffer,
            skip_k_cache_insert=self.use_fp4_kv,
            use_fp4_cache=self.use_fp4_kv,
        )

    def forward(
        self, hidden_states: torch.Tensor, qr: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        kw, _ = self.wk_weights_proj(hidden_states)
        k, weights = kw.split([self.head_dim, self.n_head], dim=-1)
        k = self.k_norm(k)

        nope_dim = self.head_dim - self.rope_dim
        q_pe, q_nope = q.split([self.rope_dim, nope_dim], dim=-1)
        k_pe, k_nope = k.split([self.rope_dim, nope_dim], dim=-1)
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        q = torch.cat([q_pe.reshape(-1, self.n_head, self.rope_dim), q_nope], -1)
        k = torch.cat([k_pe.reshape(-1, self.rope_dim), k_nope], dim=-1)
        q = hadamard_rows(q.view(-1, self.head_dim))
        k = hadamard_rows(k)

        if self.use_fp4_kv:
            q, q_scale = self._quantize_fp4(q, k)
            # Group-32 Q scales travel with Q and cannot be folded into weights.
            # The FP4 logits kernels take FP32 weights.
            weights = weights.float() * self.weights_scale
            q_with_scale = (q, q_scale)
        else:
            q, q_scale = per_token_group_quant_fp8(
                q, INDEXER_QUANT_BLOCK, column_major_scales=False, use_ue8m0=True
            )
            q = q.view(-1, self.n_head, self.head_dim)
            weights = weights * q_scale.view(-1, self.n_head) * self.weights_scale
            q_with_scale = (q, None)
        return self.indexer_op(
            hidden_states, q_with_scale, None if self.use_fp4_kv else k, weights
        )

    def _quantize_fp4(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Small row tiles keep decode Q from underfilling the GPU.
        q_values, q_scales, _ = downcast_to_mxfp(
            q,
            axis=-1,
            BLOCK_OUT_DIM=4 if q.shape[0] <= 1024 else 16,
            BLOCK_QUANT_DIM=self.head_dim,
        )
        metadata = get_forward_context().attn_metadata
        # Profiling has neither an allocated cache nor a real slot mapping.
        if isinstance(metadata, dict):
            cache = self.k_cache.kv_cache
            slots = metadata[self.k_cache.prefix].slot_mapping
            num_tokens = min(k.shape[0], slots.numel())
            if num_tokens:
                k_values, k_scales, _ = downcast_to_mxfp(
                    k[:num_tokens],
                    axis=-1,
                    BLOCK_OUT_DIM=min(triton.next_power_of_2(num_tokens), 32),
                    BLOCK_QUANT_DIM=self.head_dim,
                )
                _store_indexer_fp4[(num_tokens,)](
                    k_values,
                    k_scales,
                    cache,
                    slots,
                    k_values.stride(0),
                    k_scales.stride(0),
                    cache.stride(0),
                    PAGE_SIZE=cache.shape[1],
                    VALUE_BYTES=self.head_dim // 2,
                    SCALE_BYTES=self.head_dim // MXFP4_BLOCK_SIZE,
                    num_warps=1,
                )
        return (
            q_values.view(-1, self.n_head, self.head_dim // 2),
            q_scales.view(torch.int32).view(-1, self.n_head),
        )


class Ernie5_5MLAAttention(MLAAttention):
    """Sparse MLA layer whose KV cache uses the DSA page size."""

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return _dsa_page_spec(super().get_kv_cache_spec(vllm_config), vllm_config)


def _check_sink_backend(head_size: int, kv_cache_dtype: str) -> None:
    capability = current_platform.get_device_capability()
    if capability is None:
        raise RuntimeError("ERNIE 5.5 DSA needs a known CUDA device capability.")
    invalid_reasons = Ernie5_5FlashMLASparseBackend.validate_configuration(
        head_size=head_size,
        dtype=torch.get_default_dtype(),
        kv_cache_dtype=cast(CacheDType, kv_cache_dtype),
        block_size=DSA_BLOCK_SIZE,
        use_mla=True,
        has_sink=True,
        use_sparse=True,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=capability,
        attn_type=AttentionType.DECODER,
    )
    if invalid_reasons:
        raise RuntimeError(
            "ERNIE 5.5 DSA needs the sink-capable FLASHMLA_SPARSE backend: "
            + ", ".join(invalid_reasons)
        )


class Ernie5_5DSAAttention(nn.Module):
    """MLA layer with a DSA indexer, attention sink, VHA and an output gate."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        swa_topk_buffer: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        _check_attn_tp(config)
        quant_config = vllm_config.quant_config
        # MLAAttention may canonicalize the dtype on the config it is given.
        cache_config = copy.copy(vllm_config.cache_config)
        self.num_heads = config.mla_num_heads
        self.num_local_heads = self.num_heads // get_tensor_model_parallel_world_size()
        self.q_lora_rank = config.mla_q_lora_rank
        self.kv_lora_rank = config.mla_kv_lora_rank
        self.qk_nope_head_dim = config.mla_qk_nope_head_dim
        self.qk_rope_head_dim = config.mla_qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.mla_v_head_dim
        self.window = config.sliding_window

        self.fused_qkv_a_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_qkv_a_proj",
            disable_tp=True,
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        # Named like the HCA output projection so one mapper rule covers both.
        self.wo_b = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )
        self.o_gate = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.v_head_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.o_gate",
        )
        self.vha = VHAPostMix(self.num_heads)
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            max_position=config.max_position_embeddings,
            is_neox_style=False,
            rope_parameters=config.mla_rope_parameters,
        )
        self.attn_sink = nn.Parameter(
            torch.empty(self.num_local_heads, dtype=torch.float32), requires_grad=False
        )

        _check_sink_backend(
            self.kv_lora_rank + self.qk_rope_head_dim, cache_config.cache_dtype
        )
        # The dense MLA prefill backends cannot apply the sink.
        get_current_vllm_config().attention_config.sparse_mla_force_mqa = True

        self.indexer = Ernie5_5Indexer(
            vllm_config,
            config,
            quant_config,
            cache_config,
            topk_indices_buffer,
            swa_topk_buffer,
            prefix=f"{prefix}.indexer",
        )
        self.mla_attn = Ernie5_5MLAAttention(
            num_heads=self.num_local_heads,
            scale=self.qk_head_dim**-0.5,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            kv_b_proj=self.kv_b_proj,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_backend=Ernie5_5FlashMLASparseBackend,
            use_sparse=True,
            indexer=self.indexer,
            topk_indices_buffer=swa_topk_buffer,
            sinks=self.attn_sink,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        q_c, kv_lora = self.fused_qkv_a_proj(hidden_states)[0].split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
        )
        q_c = self.q_a_layernorm(q_c)
        gate = self.o_gate(q_c)
        q = self.q_b_proj(q_c)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], -1)
        kv_c_normed = self.kv_a_layernorm(kv_c)
        q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim :], k_pe.unsqueeze(1)
        )

        attn_out = torch.empty(
            (num_tokens, self.num_local_heads * self.v_head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self._indexer_and_attn(
            hidden_states, q_c, positions, q, kv_c_normed, k_pe, attn_out
        )
        o = self.vha(attn_out.view(num_tokens, self.num_local_heads, self.v_head_dim))
        return self.wo_b(o.flatten(1) * torch.sigmoid(gate))

    @eager_break_during_capture
    def _indexer_and_attn(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor,
        positions: torch.Tensor,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """Indexer, window merge and sparse MLA in one eager segment."""
        num_tokens = hidden_states.shape[0]
        self.indexer(hidden_states, q_c, positions)
        merge_swa_topk(
            self.indexer.indexer_op.topk_indices_buffer[:num_tokens],
            self.indexer.topk_indices_buffer[:num_tokens],
            positions[:num_tokens],
            self.window,
        )
        self.mla_attn.impl.record_logical_topk_ready()  # type: ignore[attr-defined]
        out.copy_(self.mla_attn(q, kv_c_normed, k_pe, output_shape=out.shape))
