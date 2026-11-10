# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ERNIE 5 hybrid HCA/DSA mHC decoder."""

import typing
from collections.abc import Callable, Iterable
from itertools import islice

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.kernels.mhc.tilelang import (
    hc_head_fused_kernel_tilelang,
    mhc_post_tilelang,
)
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
    spec_decode_needs_target_embed,
)
from vllm.sequence import IntermediateTensors

from vllm_ernie.models.ernie5_5.config import Ernie5_5Config
from vllm_ernie.models.ernie5_5.mhc import (
    mhc_fused_post_pre_tilelang,
    mhc_pre_broadcast_tilelang,
    mhc_pre_tilelang,
)

from .attention import Ernie5_5DSAAttention, fold_vha, select_ernie5_5_hca_cls
from .moe import Ernie5_5MLP, Ernie5_5MoE
from .quant_config import (
    dequantize_block_fp8,
    fp8_block_size,
    to_mxfp8_weight_scale,
    use_ernie5_5_mxfp4_experts,
    use_ernie5_5_mxfp8,
)


def fp32_param(*shape: int) -> nn.Parameter:
    return nn.Parameter(torch.empty(shape, dtype=torch.float32), requires_grad=False)


def topk_buffers(
    num_tokens: int, config: Ernie5_5Config
) -> tuple[torch.Tensor, torch.Tensor]:
    """Indexer top-k buffer and the wider sparse MLA buffer with the window."""
    indexer = torch.empty(num_tokens, config.index_topk, dtype=torch.int32)
    swa = torch.empty(
        num_tokens, config.index_topk + config.sliding_window, dtype=torch.int32
    )
    return indexer, swa


# Final hidden states plus the captured DSpark aux hidden states.
Ernie5_5AuxOutput = tuple[torch.Tensor, list[torch.Tensor]]


class Ernie5_5ParallelLMHead(ParallelLMHead):
    """lm_head that also owns the learned multimax applied to its logits."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer("multimax_ranges", torch.zeros(4, dtype=torch.float32))
        self.register_buffer("multimax_ts", torch.zeros(4, dtype=torch.float32))

    def multimax(self, logits: torch.Tensor) -> torch.Tensor:
        r, t = self.multimax_ranges, self.multimax_ts
        return (
            logits
            + t[0] * (r[0] - logits).relu()
            + t[1] * (logits - r[1]).relu()
            + t[2] * (r[2] - logits).relu().square()
            + t[3] * (logits - r[3]).relu().square()
        )


class Ernie5_5DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        swa_topk_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        layer_id = extract_layer_index(prefix)
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        if config.is_dsa_layer(layer_id):
            self.attn = Ernie5_5DSAAttention(
                vllm_config,
                prefix=f"{prefix}.attn",
                topk_indices_buffer=topk_indices_buffer,
                swa_topk_buffer=swa_topk_buffer,
            )
        else:
            self.attn = select_ernie5_5_hca_cls(vllm_config.cache_config.cache_dtype)(
                vllm_config,
                prefix=f"{prefix}.attn",
                topk_indices_buffer=topk_indices_buffer,
                aux_stream_list=aux_stream_list,
            )
        self.ffn: Ernie5_5MLP | Ernie5_5MoE
        if layer_id < config.first_k_dense_replace:
            self.ffn = Ernie5_5MLP(
                config,
                config.intermediate_size,
                quant_config=quant_config,
                prefix=f"{prefix}.ffn",
            )
        else:
            self.ffn = Ernie5_5MoE(
                config, quant_config=quant_config, prefix=f"{prefix}.ffn"
            )

        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = fp32_param(mix_hc, hc_dim)
        self.hc_ffn_fn = fp32_param(mix_hc, hc_dim)
        self.hc_attn_base = fp32_param(mix_hc)
        self.hc_ffn_base = fp32_param(mix_hc)
        self.hc_attn_scale = fp32_param(3)
        self.hc_ffn_scale = fp32_param(3)
        self.hc_attn_fn_broadcast: torch.Tensor | None = None

        if vllm_config.kernel_config.enable_jit_warmup:
            self._register_mhc_warmup(vllm_config, layer_id)

    def _register_mhc_warmup(self, vllm_config: VllmConfig, layer_id: int) -> None:
        from vllm.model_executor.kernels.mhc.tilelang_kernels import (
            _HC_PRENORM_GEMM_TILELANG_KERNEL,
            _MHC_FUSED_TILELANG_KERNEL,
            _MHC_POST_TILELANG_KERNEL,
        )
        from vllm.utils.deep_gemm import is_deep_gemm_supported

        from vllm_ernie.models.ernie5_5.mhc_kernels import (
            _MHC_PRE_BIG_FUSE_TILELANG_KERNEL,
        )

        include_pre_gemm_splits = is_deep_gemm_supported()
        _MHC_PRE_BIG_FUSE_TILELANG_KERNEL.register_warmup(
            vllm_config,
            hidden_size=self.hidden_size,
            hc_mult=self.hc_mult,
            use_norm_weight=True,
            include_pre_gemm_splits=include_pre_gemm_splits,
            include_broadcast_splits=get_pp_group().is_first_rank and layer_id == 0,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.hc_post_alpha,
            sinkhorn_repeat=self.hc_sinkhorn_iters,
            norm_eps=(self.attn_norm.variance_epsilon, self.ffn_norm.variance_epsilon),
            broadcast_norm_eps=self.attn_norm.variance_epsilon,
        )
        if not include_pre_gemm_splits:
            _HC_PRENORM_GEMM_TILELANG_KERNEL.register_warmup(
                vllm_config,
                hidden_size=self.hidden_size,
                hc_mult=self.hc_mult,
                n_out=self.hc_mult * (2 + self.hc_mult),
            )
        _MHC_POST_TILELANG_KERNEL.register_warmup(
            hidden_size=self.hidden_size, hc_mult=self.hc_mult
        )
        _MHC_FUSED_TILELANG_KERNEL.register_warmup(
            vllm_config, hidden_size=self.hidden_size, hc_mult=self.hc_mult
        )

    def _mhc_args(self, kind: str) -> tuple:
        return (
            getattr(self, f"hc_{kind}_fn"),
            getattr(self, f"hc_{kind}_scale"),
            getattr(self, f"hc_{kind}_base"),
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        norm_kwargs = dict(
            norm_weight=self.attn_norm.weight.data,
            norm_eps=self.attn_norm.variance_epsilon,
        )
        if residual is None:
            if x.dim() == 2:
                # First layer: expand the token embedding to hc_mult streams.
                assert self.hc_attn_fn_broadcast is not None
                residual, post_mix, res_mix, x = mhc_pre_broadcast_tilelang(
                    x,
                    *self._mhc_args("attn"),
                    fn_broadcast=self.hc_attn_fn_broadcast,
                    **norm_kwargs,
                )
            else:
                residual = x
                post_mix, res_mix, x = mhc_pre_tilelang(
                    x, *self._mhc_args("attn"), **norm_kwargs
                )
        else:
            residual, post_mix, res_mix, x = mhc_fused_post_pre_tilelang(
                x,
                residual,
                post_mix,
                res_mix,
                *self._mhc_args("attn"),
                n_splits=1,
                tile_n=1,
                **norm_kwargs,
            )

        x = self.attn(positions, x, None)

        residual, post_mix, res_mix, x = mhc_fused_post_pre_tilelang(
            x,
            residual,
            post_mix,
            res_mix,
            *self._mhc_args("ffn"),
            n_splits=1,
            tile_n=1,
            norm_weight=self.ffn_norm.weight.data,
            norm_eps=self.ffn_norm.variance_epsilon,
        )
        x = self.ffn(x)
        return x, residual, post_mix, res_mix


class Ernie5_5Model(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.mxfp8 = use_ernie5_5_mxfp8(vllm_config)
        self.mxfp4_experts = use_ernie5_5_mxfp4_experts(vllm_config)
        self.fp8_block = fp8_block_size(vllm_config)
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens

        # One aux stream per non-default HCA input GEMM.
        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]
        # Shared by all DSA layers: the indexer output, and the sparse MLA
        # indices with the window appended. Not named `topk_indices_buffer`:
        # the MTP proposer would rebind both draft buffers to that one tensor.
        self.indexer_topk_buffer, self.swa_topk_buffer = topk_buffers(
            max_num_tokens, config
        )

        if get_pp_group().is_first_rank or spec_decode_needs_target_embed(vllm_config):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Ernie5_5DecoderLayer(
                vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.indexer_topk_buffer,
                swa_topk_buffer=self.swa_topk_buffer,
                aux_stream_list=aux_stream_list,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.hc_head_fn = fp32_param(self.hc_mult, self.hc_dim)
        self.hc_head_base = fp32_param(self.hc_mult)
        self.hc_head_scale = fp32_param(1)

        spec_config = vllm_config.speculative_config
        self._mtp_hidden_buffer: torch.Tensor | None = None
        if (
            get_pp_group().is_last_rank
            and spec_config is not None
            and (spec_config.use_eagle() or spec_config.uses_draft_model())
        ):
            self._mtp_hidden_buffer = torch.empty(
                max_num_tokens, self.hc_dim, dtype=vllm_config.model_config.dtype
            )

        if vllm_config.kernel_config.enable_jit_warmup and get_pp_group().is_last_rank:
            from vllm.model_executor.kernels.mhc.tilelang_kernels import (
                _HC_HEAD_FUSED_TILELANG_KERNEL,
            )

            _HC_HEAD_FUSED_TILELANG_KERNEL.register_warmup(
                hidden_size=config.hidden_size,
                hc_mult=self.hc_mult,
                rms_eps=self.rms_norm_eps,
                hc_eps=self.hc_eps,
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.hc_mult, self.config.hidden_size),
                    dtype=dtype,
                    device=device,
                ),
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | Ernie5_5AuxOutput:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        residual, post_mix, res_mix = None, None, None
        aux_hidden_states: list[torch.Tensor] = []
        aux_recon: torch.Tensor | None = None
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states, positions, post_mix, res_mix, residual
            )
            if idx + 1 in self.aux_hidden_state_layers:
                # DSpark drafts consume the stream-averaged mHC output.
                aux_recon = mhc_post_tilelang(
                    hidden_states, residual, post_mix, res_mix
                )
                aux_hidden_states.append(aux_recon.mean(dim=1))
        if residual is not None:
            if self.end_layer in self.aux_hidden_state_layers:
                assert aux_recon is not None
                hidden_states = aux_recon
            else:
                hidden_states = mhc_post_tilelang(
                    hidden_states, residual, post_mix, res_mix
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        if self._mtp_hidden_buffer is not None:
            num_tokens = hidden_states.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))
        hidden_states = hc_head_fused_kernel_tilelang(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        hidden_states = self.norm(hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        num_layers = self.config.num_hidden_layers

        # The trailing layers belong to the MTP draft model, `mtp.*` to a
        # bundled draft model.
        def is_target(name: str) -> bool:
            if name.startswith("mtp."):
                return False
            return not name.startswith("layers.") or (
                int(name.split(".")[1]) < num_layers
            )

        weights = ((name, w) for name, w in weights if is_target(name))
        return load_ernie5_5_weights(self, weights, self.get_expert_mapping())

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return ernie5_5_expert_mapping(self, self.config)

    def finalize_mhc_broadcast_weights(self) -> None:
        if not get_pp_group().is_first_rank or self.start_layer >= self.end_layer:
            return
        layer = self.layers[self.start_layer]
        broadcast = (
            layer.hc_attn_fn.detach()
            .view(-1, layer.hc_mult, layer.hidden_size)
            .sum(dim=1)
        )
        if layer.hc_attn_fn_broadcast is None:
            layer.hc_attn_fn_broadcast = broadcast
        else:
            layer.hc_attn_fn_broadcast.copy_(broadcast)


def ernie5_5_expert_mapping(
    model: nn.Module, config: Ernie5_5Config
) -> list[tuple[str, str, int, str]]:
    return fused_moe_make_expert_params_mapping(
        model,
        ckpt_gate_proj_name="w1",
        ckpt_down_proj_name="w2",
        ckpt_up_proj_name="w3",
        num_experts=config.n_routed_experts,
    )


def load_ernie5_5_weights(
    model: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    expert_mapping: list[tuple[str, str, int, str]],
) -> set[str]:
    """Load mapper-renamed ERNIE 5 weights whose names are relative to `model`.

    Stacked shards arrive with the `shard_id` attribute set by `ernie5_5_mapper`.
    With `model.mxfp8`, block-FP8 ``weight_scale_inv`` is loaded as MXFP8
    ``weight_scale``. FP8 weights of bf16 layers are dequantized.
    """
    params_dict = dict(model.named_parameters())
    loaded_params: set[str] = set()
    mxfp8 = getattr(model, "mxfp8", False)
    mxfp4_experts = getattr(model, "mxfp4_experts", False)
    pending: dict[tuple[str, int | None], list[torch.Tensor | None]] = {}
    block = getattr(model, "fp8_block", [128, 128])
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    for name, loaded_weight in _dequantize_bf16_layers(
        weights, params_dict, pending, block
    ):
        if is_pp_missing_parameter(name, model):
            continue
        shard_id = getattr(loaded_weight, "shard_id", None)
        if mxfp8 and name.endswith(".weight_scale_inv"):
            name = name.removesuffix("_inv")
            loaded_weight = to_mxfp8_weight_scale(loaded_weight)
        if ".experts." in name:
            if mxfp4_experts:
                # Packed FP4 pairs (int8) and e8m0 scales for `Mxfp4MoEMethod`.
                name = name.removesuffix("_inv")
                if loaded_weight.dtype == torch.int8:
                    loaded_weight = loaded_weight.view(torch.uint8)
            for param_name, weight_name, expert_id, expert_shard_id in expert_mapping:
                if weight_name not in name:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                param = params_dict[name_mapped]
                weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
                # Experts owned by other EP ranks report failure.
                if weight_loader(
                    param,
                    loaded_weight,
                    name_mapped,
                    shard_id=expert_shard_id,
                    expert_id=expert_id,
                    return_success=True,
                ):
                    loaded_params.add(name_mapped)
                    break
            continue

        if name not in params_dict:
            head, _, leaf = name.rpartition(".")
            name = f"{head}.base_layer.{leaf}"
        param = params_dict[name]
        leaf = name.rpartition(".")[2]
        if leaf.startswith("hc_") and leaf.endswith("_fn"):
            # mHC mixers are stored as [in, out].
            param.data.copy_(loaded_weight.t())
        elif leaf in ("hc_attn_scale", "hc_ffn_scale"):
            param.data[shard_id : shard_id + 1].copy_(loaded_weight)
        elif name.endswith(".ffn.gate.weight"):
            # The two split-feature routers share one [2E, H] GEMM.
            rows = loaded_weight.shape[0]
            start = (shard_id or 0) * rows
            param.data[start : start + rows].copy_(loaded_weight)
        elif leaf == "attn_sink":
            # This rank's heads; HCA pads the param past them.
            n = loaded_weight.shape[0] // tp_size
            param.data[:n].copy_(loaded_weight[tp_rank * n : (tp_rank + 1) * n])
        elif shard_id is not None:
            param.weight_loader(param, loaded_weight, shard_id)
        else:
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
        loaded_params.add(name)
    if pending:
        raise ValueError(f"FP8 weights without scales: {sorted(pending)[:4]}")
    return loaded_params


def _dequantize_bf16_layers(
    weights: Iterable[tuple[str, torch.Tensor]],
    params_dict: dict[str, nn.Parameter],
    pending: dict[tuple[str, int | None], list[torch.Tensor | None]],
    block: list[int],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Pair block-FP8 weights of bf16-built layers with their scales."""

    def bf16_param(weight_name: str) -> bool:
        head, _, leaf = weight_name.rpartition(".")
        param = params_dict.get(
            weight_name, params_dict.get(f"{head}.base_layer.{leaf}")
        )
        return param is not None and param.dtype != torch.float8_e4m3fn

    for name, loaded_weight in weights:
        if name.endswith(".weight_scale_inv"):
            weight_name, slot = name.removesuffix("_scale_inv"), 1
        elif loaded_weight.dtype == torch.float8_e4m3fn:
            weight_name, slot = name, 0
        else:
            yield name, loaded_weight
            continue
        if not bf16_param(weight_name):
            yield name, loaded_weight
            continue
        shard_id = getattr(loaded_weight, "shard_id", None)
        pair = pending.setdefault((weight_name, shard_id), [None, None])
        pair[slot] = loaded_weight
        if pair[0] is None or pair[1] is None:
            continue
        del pending[(weight_name, shard_id)]
        weight = dequantize_block_fp8(pair[0], pair[1], block)
        if shard_id is not None:
            weight.shard_id = shard_id  # type: ignore[attr-defined]
        yield weight_name, weight


def _hc_rules(ckpt: str, kind: str) -> dict[str, tuple[str, int]]:
    return {
        f".{ckpt}_hyper_connection.alpha_{alpha}": (f".hc_{kind}_scale", i)
        for i, alpha in enumerate(("pre", "post", "res"))
    }


# Every pattern spans whole dotted segments so that e.g. `.wk.` cannot match
# inside `linear_wkv`. Substring rules apply in order, first match only.
ernie5_5_mapper = WeightsMapper(
    orig_to_new_substr={
        ".transformer_layer.": ".",
        ".self_attn.core_attention.softmax_offset": ".attn.attn_sink",
        ".self_attn.core_attention.": ".attn.",
        ".self_attn.gate_proj.": ".attn.o_gate.",
        ".self_attn.vha_postmix_": ".attn.vha.",
        ".self_attn.q_layernorm.": ".attn.q_norm.",
        ".self_attn.kv_layernorm.": ".attn.kv_norm.",
        ".self_attn.linear_q_up_proj.": ".attn.wq_b.",
        ".self_attn.linear_o_group_proj": ".attn.wo_a",
        ".self_attn.o_proj.": ".attn.wo_b.",
        ".self_attn.": ".attn.",
        ".block_sparse_moe.": ".ffn.",
        ".mlp.": ".ffn.",
        ".shared_experts.w2.": ".shared_experts.down_proj.",
        ".input_layernorm.": ".attn_norm.",
        ".post_attention_layernorm.": ".ffn_norm.",
        ".self_attention_hyper_connection.mapping_proj.weight": ".hc_attn_fn",
        ".self_attention_hyper_connection.bias": ".hc_attn_base",
        ".mlp_hyper_connection.mapping_proj.weight": ".hc_ffn_fn",
        ".mlp_hyper_connection.bias": ".hc_ffn_base",
    },
    orig_to_new_stacked={
        ".attn.linear_q_down_proj.": (".attn.fused_wqa_wkv.", 0),
        ".attn.linear_kv_proj.": (".attn.fused_wqa_wkv.", 1),
        ".attn.q_a_proj.": (".attn.fused_qkv_a_proj.", 0),
        ".attn.kv_a_proj_with_mqa.": (".attn.fused_qkv_a_proj.", 1),
        ".compressor.linear_wkv.": (".compressor.fused_wkv_wgate.", 0),
        ".compressor.linear_wgate.": (".compressor.fused_wkv_wgate.", 1),
        ".indexer.wk.": (".indexer.wk_weights_proj.", 0),
        ".indexer.weights_proj.": (".indexer.wk_weights_proj.", 1),
        ".shared_experts.w1.": (".shared_experts.gate_up_proj.", 0),
        ".shared_experts.w3.": (".shared_experts.gate_up_proj.", 1),
        ".ffn.gate_proj.": (".ffn.gate_up_proj.", 0),
        ".ffn.up_proj.": (".ffn.gate_up_proj.", 1),
        ".ffn.gate.weight_1": (".ffn.gate.weight", 1),
        **_hc_rules("self_attention", "attn"),
        **_hc_rules("mlp", "ffn"),
    },
    orig_to_new_prefix={
        "model.mhc_contract.": "model.",
        "model.lm_head.": "lm_head.",
    },
    orig_to_new_suffix={
        ".attn.wo_a": ".attn.wo_a.weight",
        ".attn.wo_a_scale_inv": ".attn.wo_a.weight_scale_inv",
    },
)


class Ernie5_5MixtureOfExperts(MixtureOfExperts):
    moe_mlp_layers: list[Ernie5_5MoE]

    def extract_moe_parameters(self, example_moe: Ernie5_5MoE | None) -> None:
        if example_moe is None:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
            return
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_shared_experts = example_moe.n_shared_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self, num_physical_experts: int, num_local_physical_experts: int
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


class Ernie5_5ForCausalLM(
    nn.Module, SupportsPP, SupportsEagle3, Ernie5_5MixtureOfExperts
):
    hf_to_vllm_mapper = ernie5_5_mapper

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_wqa_wkv": ["linear_q_down_proj", "linear_kv_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
        "fused_wkv_wgate": ["linear_wkv", "linear_wgate"],
        "wk_weights_proj": ["wk", "weights_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = Ernie5_5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = Ernie5_5ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
        self.num_expert_groups = 1
        self.moe_layers: list[nn.Module] = []
        self.moe_mlp_layers: list[Ernie5_5MoE] = []
        for layer in self.model.layers:
            if isinstance(layer, Ernie5_5DecoderLayer) and isinstance(
                layer.ffn, Ernie5_5MoE
            ):
                self.moe_mlp_layers.append(layer.ffn)
                self.moe_layers.append(layer.ffn.experts)
        self.num_moe_layers = len(self.moe_layers)
        self.extract_moe_parameters(
            self.moe_mlp_layers[0] if self.moe_mlp_layers else None
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | Ernie5_5AuxOutput:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is None:
            return None
        return self.lm_head.multimax(logits)

    def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states, skip_gather=True)
        return self.lm_head.multimax(logits)

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        return self.model._mtp_hidden_buffer

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def process_weights_after_loading(self) -> None:
        fold_vha(self)
        self.model.finalize_mhc_broadcast_weights()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
