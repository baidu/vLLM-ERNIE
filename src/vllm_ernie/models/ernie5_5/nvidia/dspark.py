# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark draft model for ERNIE 5 (NVIDIA): SWA-only mHC stages + Markov head."""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.tilelang import (
    hc_head_fused_kernel_tilelang,
    mhc_post_tilelang,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen3_dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope
from vllm.models.deepseek_v4.nvidia.dspark import _insert_context_kv

from .attention import fold_vha
from .model import (
    Ernie5_5DecoderLayer,
    Ernie5_5ParallelLMHead,
    ernie5_5_expert_mapping,
    ernie5_5_mapper,
    fp32_param,
    load_ernie5_5_weights,
    topk_buffers,
)
from .quant_config import use_ernie5_5_mxfp4_experts

logger = init_logger(__name__)

# Draft-level checkpoint weights renamed outside `ernie5_5_mapper`.
_DSPARK_RENAMES = {
    "model.feature_projection.weight": "model.main_proj.weight",
    "model.feature_projection.weight_scale_inv": "model.main_proj.weight_scale_inv",
    "model.feature_norm.weight": "model.main_norm.weight",
    "model.confidence_head.projection.weight": "model.confidence_head.proj.weight",
}
# Shared from the target after loading.
_TARGET_OWNED = ("lm_head.", "model.embed_tokens.", "model.lm_head.")
_BUNDLED_PREFIX = "model.mtp."


def _unbundle_dspark_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Rename a draft bundled under `model.mtp.*` to the standalone export
    layout and drop the target's weights, whose names would clash."""
    for name, loaded_weight in weights:
        rest = name.removeprefix(_BUNDLED_PREFIX)
        if rest == name:
            continue
        if rest.partition(".")[0].isdigit():
            yield f"model.layers.{rest}", loaded_weight
        else:
            yield f"model.{rest}", loaded_weight


class Ernie5_5DSparkModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        # The stages reuse the target decoder, so they read its config.
        target_config = vllm_config.model_config.hf_config
        self.config = config
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.num_dspark_layers = config.num_hidden_layers
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.main_proj = ReplicatedLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            # FP8 in MXFP4-expert checkpoints, BF16 in the others.
            quant_config=(
                vllm_config.quant_config
                if use_ernie5_5_mxfp4_experts(vllm_config)
                else None
            ),
            prefix=maybe_prefix(prefix, "main_proj"),
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # After all target layers: unique names, uncompressed SWA attention.
        self.layer_offset = (
            target_config.num_hidden_layers + target_config.num_nextn_predict_layers
        )
        self.indexer_topk_buffer, self.swa_topk_buffer = topk_buffers(
            vllm_config.scheduler_config.max_num_batched_tokens, target_config
        )
        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]
        self.layers = nn.ModuleList(
            [
                Ernie5_5DecoderLayer(
                    vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{self.layer_offset + i}"),
                    topk_indices_buffer=self.indexer_topk_buffer,
                    swa_topk_buffer=self.swa_topk_buffer,
                    aux_stream_list=aux_stream_list,
                )
                for i in range(self.num_dspark_layers)
            ]
        )
        for layer in self.layers:
            attn = layer.attn
            # Trained with plain RoPE at rope_theta (no YaRN), like SWA layers.
            attn.rotary_emb = attn.indexer_rotary_emb = build_deepseek_v4_rope(
                config,
                head_dim=attn.head_dim,
                rope_head_dim=attn.rope_head_dim,
                max_position_embeddings=target_config.max_position_embeddings,
                compress_ratio=1,
            )
        self.context_wkv_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.head_dim] * self.num_dspark_layers,
            bias=False,
            return_bias=False,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "context_wkv_proj"),
            disable_tp=True,
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = fp32_param(self.hc_mult, hc_dim)
        self.hc_head_base = fp32_param(self.hc_mult)
        self.hc_head_scale = fp32_param(1)
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            config.vocab_size,
            config.dspark_markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )
        self.confidence_head: DSparkConfidenceHead | None = None
        if getattr(config, "enable_confidence_head", False):
            self.confidence_head = DSparkConfidenceHead(
                config.hidden_size + config.dspark_markov_rank,
                prefix=maybe_prefix(prefix, "confidence_head"),
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.main_norm(self.main_proj(aux_hidden_states))

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        main_x: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
    ) -> None:
        all_kv = self.context_wkv_proj(main_x).view(
            main_x.shape[0], self.num_dspark_layers, self.config.head_dim
        )
        for i, (layer, kv) in enumerate(
            zip(self.layers, all_kv.unbind(1), strict=True)
        ):
            slot_mapping = (
                None if context_slot_mappings is None else context_slot_mappings[i]
            )
            attn = layer.attn
            kv = attn.kv_norm(kv)
            if slot_mapping is None:
                continue
            _insert_context_kv(attn, kv, context_positions, slot_mapping)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        residual = post_mix = res_mix = None
        for layer in self.layers:
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states, positions, post_mix, res_mix, residual
            )
        hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)
        return hc_head_fused_kernel_tilelang(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )


class Ernie5_5DSparkForCausalLM(nn.Module):
    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.mxfp4_experts = use_ernie5_5_mxfp4_experts(vllm_config)
        self.model = Ernie5_5DSparkModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # Replaced by the target's lm_head after loading.
        self.lm_head = Ernie5_5ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(aux_hidden_states)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.attn.swa_cache_layer.prefix for layer in self.model.layers]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mappings
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # No multimax, like SGLang: the target's is non-monotonic past a
        # ckpt-dependent logit and demotes confident draft tokens.
        return self.logits_processor(self.lm_head, self.model.norm(hidden_states))

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.compute_logits(hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def compute_confidence(
        self, head_hidden: torch.Tensor, markov_embed: torch.Tensor
    ) -> torch.Tensor:
        assert self.model.confidence_head is not None
        return torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))

    def _map_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, loaded_weight in weights:
            if name.startswith(_TARGET_OWNED):
                continue
            if name in _DSPARK_RENAMES:
                name = _DSPARK_RENAMES[name]
                if "confidence_head." in name and self.model.confidence_head is None:
                    continue
                yield name, loaded_weight
                continue
            if name == "model.markov_head.markov_w1.weight":
                # Stored as [rank, vocab]; nn.Embedding wants [vocab, rank].
                yield name, loaded_weight.t()
                continue
            if ".self_attn.linear_kv_proj." in name:
                # Each stage's context KV comes from the shared projection;
                # keep the leaf so FP8 `weight_scale_inv` loads alongside.
                layer_idx, _, leaf = name.removeprefix("model.layers.").partition(
                    ".self_attn.linear_kv_proj."
                )
                context_weight = loaded_weight.detach()
                context_weight.shard_id = int(layer_idx)
                yield f"model.context_wkv_proj.{leaf}", context_weight
            mapped = ernie5_5_mapper._map_name_with_shard(name)
            if mapped is None:
                continue
            name, shard_id = mapped
            if shard_id is not None:
                loaded_weight.shard_id = shard_id
            yield name, loaded_weight

    def is_unused_checkpoint_weight(self, name: str) -> bool:
        # Lets the loader skip the bundling target's shards without draft weights.
        bundled = getattr(self.config, "dspark_bundled_in_target", False)
        return bundled and not name.startswith(_BUNDLED_PREFIX)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if getattr(self.config, "dspark_bundled_in_target", False):
            weights = _unbundle_dspark_weights(weights)
        loaded_params = load_ernie5_5_weights(
            self,
            self._map_weights(weights),
            ernie5_5_expert_mapping(self, self.config),
        )
        if self.model.confidence_head is not None and not any(
            "confidence_head." in name for name in loaded_params
        ):
            self.model.confidence_head = None
        # Replaced by the target's modules, so nothing to load for them.
        loaded_params.update(
            name
            for name, _ in self.named_parameters()
            if name.startswith(("lm_head.", "model.embed_tokens."))
        )
        logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
        return loaded_params

    def process_weights_after_loading(self) -> None:
        fold_vha(self)
