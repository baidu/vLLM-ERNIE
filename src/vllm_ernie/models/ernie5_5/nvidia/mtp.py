# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MTP draft model for ERNIE 5."""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.kernels.mhc.tilelang import (
    hc_head_fused_kernel_tilelang,
    mhc_post_tilelang,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.mtp_validation import (
    is_mtp_completeness_check_enabled,
)
from vllm.model_executor.models.deepseek_mtp import SharedHead
from vllm.model_executor.models.utils import (
    get_spec_layer_idx_from_weight_name,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from .attention import fold_vha
from .model import (
    Ernie5_5DecoderLayer,
    ernie5_5_expert_mapping,
    ernie5_5_mapper,
    fp32_param,
    load_ernie5_5_weights,
    topk_buffers,
)
from .quant_config import fp8_block_size, use_ernie5_5_mxfp8

# Per-MTP-layer modules outside the decoder block.
_MTP_LAYER_OWN = ("enorm.", "hnorm.", "e_proj.", "h_proj.", "shared_head.", "hc_head_")


class Ernie5_5MTPLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        topk_indices_buffer: torch.Tensor,
        swa_topk_buffer: torch.Tensor,
        prefix: str,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
    ) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        quant_config = vllm_config.quant_config
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * self.hidden_size

        # e_proj / h_proj are bf16 in every checkpoint.
        self.enorm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.hnorm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.e_proj = ReplicatedLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=None,
            prefix=f"{prefix}.e_proj",
        )
        self.h_proj = ReplicatedLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=None,
            prefix=f"{prefix}.h_proj",
        )
        self.hc_head_fn = fp32_param(self.hc_mult, self.hc_dim)
        self.hc_head_base = fp32_param(self.hc_mult)
        self.hc_head_scale = fp32_param(1)
        # The multimax lm_head is shared from the target model.
        self.shared_head = SharedHead(
            config=config,
            prefix=prefix,
            quant_config=quant_config,
            defer_lm_head=True,
        )
        self.mtp_block = Ernie5_5DecoderLayer(
            vllm_config,
            prefix,
            topk_indices_buffer=topk_indices_buffer,
            swa_topk_buffer=swa_topk_buffer,
            aux_stream_list=aux_stream_list,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        # The target stashes the pre-hc_head residual flat as (T, hc_mult * D).
        previous_hidden_states = previous_hidden_states.view(
            -1, self.hc_mult, self.hidden_size
        )
        hidden_states = self.h_proj(self.hnorm(previous_hidden_states)) + self.e_proj(
            self.enorm(inputs_embeds)
        ).unsqueeze(-2)
        hidden_states, residual, post_mix, res_mix = self.mtp_block(
            x=hidden_states, positions=positions
        )
        hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)
        # Re-fed as `previous_hidden_states` by later spec steps.
        return hidden_states.flatten(1)


class Ernie5_5MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        # Own buffers, not shared with the target (see `Ernie5_5Model`).
        self.indexer_topk_buffer, self.swa_topk_buffer = topk_buffers(
            vllm_config.scheduler_config.max_num_batched_tokens, config
        )
        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]
        self.layers = nn.ModuleDict(
            {
                str(idx): Ernie5_5MTPLayer(
                    vllm_config,
                    self.indexer_topk_buffer,
                    self.swa_topk_buffer,
                    f"{prefix}.layers.{idx}",
                    aux_stream_list=aux_stream_list,
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _layer(self, spec_step_idx: int) -> Ernie5_5MTPLayer:
        step = spec_step_idx % self.num_mtp_layers
        return self.layers[str(self.mtp_start_layer_idx + step)]  # type: ignore[return-value]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        return self._layer(spec_step_idx)(
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            spec_step_idx % self.num_mtp_layers,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor | None:
        layer = self._layer(spec_step_idx)
        hidden_states = hc_head_fused_kernel_tilelang(
            hidden_states.view(-1, layer.hc_mult, layer.hidden_size),
            layer.hc_head_fn,
            layer.hc_head_scale,
            layer.hc_head_base,
            layer.rms_norm_eps,
            layer.hc_eps,
        )
        hidden_states = layer.shared_head.norm(hidden_states)
        head = layer.shared_head.head
        logits = self.logits_processor(head, hidden_states)
        if logits is None:
            return None
        return head.multimax(logits)


class Ernie5_5MTP(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.mxfp8 = use_ernie5_5_mxfp8(vllm_config)
        self.fp8_block = fp8_block_size(vllm_config)
        self.model = Ernie5_5MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def _map_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, loaded_weight in weights:
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
                # The lm_head is shared from the target after loading.
                if name == "model.embed_tokens.weight":
                    yield name, loaded_weight
                continue
            mapped = ernie5_5_mapper._map_name_with_shard(name)
            if mapped is None:
                continue
            name, shard_id = mapped
            if shard_id is not None:
                loaded_weight.shard_id = shard_id
            layer_prefix = f"model.layers.{spec_layer}."
            rest = name.removeprefix(layer_prefix)
            if not rest.startswith(_MTP_LAYER_OWN):
                name = f"{layer_prefix}mtp_block.{rest}"
            yield name, loaded_weight

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_params = load_ernie5_5_weights(
            self,
            self._map_weights(weights),
            ernie5_5_expert_mapping(self, self.config),
        )
        loaded_layers = {
            get_spec_layer_idx_from_weight_name(self.config, name)
            for name in loaded_params
        }
        for layer_idx in self.model.layers:
            if int(layer_idx) not in loaded_layers and (
                is_mtp_completeness_check_enabled()
            ):
                raise ValueError(
                    f"MTP layer {layer_idx} weights are missing from the "
                    "checkpoint; disable speculative decoding or use a "
                    "checkpoint that includes them."
                )
        return loaded_params

    def process_weights_after_loading(self) -> None:
        fold_vha(self)
