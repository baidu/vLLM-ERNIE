# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers import PretrainedConfig


class Ernie5_5Config(PretrainedConfig):
    """Config for ERNIE 5 hybrid MLA (`Ernie5_5_MoeForCausalLM`, `ernie5_5_moe`).

    Layers with ``compress_ratios[i] > 1`` are DeepSeek-V4 style HCA layers
    (top-level attention fields); layers with ``compress_ratios[i] <= 0`` are
    DSA layers configured by the ``hybrid_mla_*`` fields. The trailing
    ``num_nextn_predict_layers`` entries of ``compress_ratios`` describe the
    MTP layers.
    """

    model_type = "ernie5_5_moe"

    def __init__(
        self,
        vocab_size: int = 201216,
        hidden_size: int = 4096,
        num_hidden_layers: int = 43,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 1,
        head_dim: int = 512,
        q_lora_rank: int = 1024,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 448,
        qk_rope_head_dim: int = 64,
        o_groups: int = 8,
        o_lora_rank: int = 1024,
        sliding_window: int = 128,
        compress_ratios: list[int] | None = None,
        hybrid_mla_num_attention_heads: int = 64,
        hybrid_mla_q_lora_rank: int = 1024,
        hybrid_mla_kv_lora_rank: int = 512,
        hybrid_mla_qk_nope_head_dim: int = 192,
        hybrid_mla_qk_rope_head_dim: int = 64,
        hybrid_mla_v_head_dim: int = 256,
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 2048,
        max_position_embeddings: int = 200192,
        rope_theta: float = 2500000.0,
        compress_rope_theta: float = 160000.0,
        rope_scaling: dict[str, Any] | None = None,
        rope_parameters: dict[str, Any] | None = None,
        rms_norm_eps: float = 1e-6,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        intermediate_size: int = 14336,
        moe_intermediate_size: int = 2048,
        moe_latent_size: int = 2048,
        n_routed_experts: int = 512,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 10,
        first_k_dense_replace: int = 1,
        routed_scaling_factor: float = 3.0,
        norm_topk_prob: bool = True,
        hidden_act: str = "situ",
        activation_situ_beta: float = 4.0,
        activation_situ_linear_beta: float = 25.0,
        multimax: str | None = "lm_head",
        num_nextn_predict_layers: int = 1,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.o_groups = o_groups
        self.o_lora_rank = o_lora_rank
        self.sliding_window = sliding_window
        self.compress_ratios = (
            compress_ratios
            if compress_ratios is not None
            else [128] * (num_hidden_layers + num_nextn_predict_layers)
        )
        self.hybrid_mla_num_attention_heads = hybrid_mla_num_attention_heads
        self.hybrid_mla_q_lora_rank = hybrid_mla_q_lora_rank
        self.hybrid_mla_kv_lora_rank = hybrid_mla_kv_lora_rank
        self.hybrid_mla_qk_nope_head_dim = hybrid_mla_qk_nope_head_dim
        self.hybrid_mla_qk_rope_head_dim = hybrid_mla_qk_rope_head_dim
        self.hybrid_mla_v_head_dim = hybrid_mla_v_head_dim
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.compress_rope_theta = compress_rope_theta
        # Consumed by the HCA layers only; DSA layers use mla_rope_parameters.
        self.rope_parameters = (
            rope_scaling
            or rope_parameters
            or {
                "rope_type": "default",
                "rope_theta": rope_theta,
            }
        )
        self.rms_norm_eps = rms_norm_eps
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_latent_size = moe_latent_size
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.hidden_act = hidden_act
        self.activation_situ_beta = activation_situ_beta
        self.activation_situ_linear_beta = activation_situ_linear_beta
        self.multimax = multimax
        self.num_nextn_predict_layers = num_nextn_predict_layers
        if dspark := kwargs.get("dspark_draft"):
            # DSpark draft checkpoint: expose the fields the speculator reads.
            self.dspark_target_layer_ids = dspark["feature_layer_ids"]
            self.dspark_markov_rank = dspark.get("markov_rank", 0)
            self.enable_confidence_head = dspark.get("enable_confidence_head", False)
            kwargs.setdefault("mask_token_id", dspark["mask_token_id"])
        # The trainer computes logits (and multimax) in fp32.
        kwargs.setdefault("head_dtype", "float32")
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def is_dsa_layer(self, layer_id: int) -> bool:
        # Layers past `compress_ratios` (DSpark draft layers) are SWA-only.
        return (
            layer_id < len(self.compress_ratios) and self.compress_ratios[layer_id] <= 0
        )

    # DSA layer dims, kept apart from the HCA fields of the same names.
    @property
    def mla_num_heads(self) -> int:
        return self.hybrid_mla_num_attention_heads

    @property
    def mla_q_lora_rank(self) -> int:
        return self.hybrid_mla_q_lora_rank

    @property
    def mla_kv_lora_rank(self) -> int:
        return self.hybrid_mla_kv_lora_rank

    @property
    def mla_qk_nope_head_dim(self) -> int:
        return self.hybrid_mla_qk_nope_head_dim

    @property
    def mla_qk_rope_head_dim(self) -> int:
        return self.hybrid_mla_qk_rope_head_dim

    @property
    def mla_v_head_dim(self) -> int:
        return self.hybrid_mla_v_head_dim

    @property
    def mla_rope_parameters(self) -> dict[str, Any]:
        # YaRN is HCA-only (at compress_rope_theta); DSA, its indexer and MTP
        # use plain RoPE at rope_theta.
        return {"rope_type": "default", "rope_theta": self.rope_theta}
