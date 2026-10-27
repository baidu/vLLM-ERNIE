# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense FFN and latent MoE blocks for ERNIE 5 (NVIDIA)."""

from functools import partial

import torch
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.distributed import get_ep_group
from vllm.model_executor.layers.activation import SituAndMul
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig

from vllm_ernie.models.ernie5_5.config import Ernie5_5Config


class Ernie5_5MLP(nn.Module):
    """SiTU-GLU feed-forward block (dense layers and shared experts)."""

    def __init__(
        self,
        config: Ernie5_5Config,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if config.hidden_act != "situ":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. Only situ is supported."
            )
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SituAndMul(
            beta=config.activation_situ_beta,
            linear_beta=config.activation_situ_linear_beta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


def ernie5_5_routing(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    *,
    bias: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bias-corrected top-k over pre-reduced split-feature sigmoid scores.

    ``gating_output`` already holds ``sigmoid(l0) + sigmoid(l1)``. The selected
    scores are renormalized and multiplied by the learnable per-expert scale,
    indexed by logical expert id.
    """
    ids = torch.topk(gating_output + bias, topk, dim=-1, sorted=False).indices
    weights = gating_output.gather(1, ids)
    if renormalize:
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-20)
    weights = weights * scale[ids]
    return weights.float(), ids.int()


class Ernie5_5RoutedOutputTransform(nn.Module):
    """Latent RMSNorm + up-projection applied after the expert combine."""

    def __init__(self, norm: RMSNorm, up_proj: ReplicatedLinear) -> None:
        super().__init__()
        self.norm = norm
        self.up_proj = up_proj

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.up_proj(self.norm(hidden_states))
        return hidden_states


class Ernie5_5MoE(nn.Module):
    """Latent MoE with split-feature routing and per-expert scaling.

    Two routers score every token (``gate.weight`` and ``gate.weight_1``, fused
    here into one ``[2E, H]`` GEMM). Experts run in the ``moe_latent_size``
    space between ``fc1_latent_proj`` and ``latent_norm`` + ``fc2_latent_proj``;
    the shared expert sees the original hidden states.
    """

    def __init__(
        self,
        config: Ernie5_5Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        if not getattr(config, "moe_split_feature_routing", False):
            raise ValueError("ERNIE 5.5 MoE requires moe_split_feature_routing.")
        if not getattr(config, "routed_scaling_factor_learnable", False):
            raise ValueError("ERNIE 5.5 MoE requires routed_scaling_factor_learnable.")
        if getattr(config, "n_group", 1) != 1 or getattr(config, "topk_group", 1) != 1:
            raise NotImplementedError("ERNIE 5.5 MoE only supports one expert group.")
        if getattr(config, "scoring_func", "sigmoid") != "sigmoid":
            raise NotImplementedError("ERNIE 5.5 MoE only supports sigmoid scoring.")

        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config
        if parallel_config.enable_eplb:
            # The per-expert scale is indexed by logical expert id.
            raise NotImplementedError("ERNIE 5.5 MoE does not support EPLB yet.")

        hidden_size = config.hidden_size
        latent_size = config.moe_latent_size
        num_experts = config.n_routed_experts
        self.n_routed_experts = num_experts
        self.n_shared_experts = config.n_shared_experts
        self.ep_size = get_ep_group().world_size
        self.ep_rank = get_ep_group().rank_in_group
        self.n_logical_experts = num_experts
        self.n_redundant_experts = 0
        self.n_physical_experts = num_experts
        self.n_local_physical_experts = num_experts // self.ep_size
        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.gate = GateLinear(
            hidden_size,
            2 * num_experts,
            bias=False,
            out_dtype=torch.float32,
            params_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.gate.routed_scaling_factor_param = nn.Parameter(
            torch.ones(num_experts, dtype=torch.float32), requires_grad=False
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(num_experts, dtype=torch.float32), requires_grad=False
        )

        self.shared_experts = Ernie5_5MLP(
            config,
            config.moe_intermediate_size * config.n_shared_experts,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.shared_experts",
        )
        self.fc1_latent_proj = ReplicatedLinear(
            hidden_size,
            latent_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1_latent_proj",
        )
        self.latent_norm = RMSNorm(latent_size, eps=config.rms_norm_eps)
        self.fc2_latent_proj = ReplicatedLinear(
            latent_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2_latent_proj",
        )

        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            num_experts=num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=latent_size,
            intermediate_size=config.moe_intermediate_size,
            activation=config.hidden_act,
            activation_situ_beta=config.activation_situ_beta,
            activation_situ_linear_beta=config.activation_situ_linear_beta,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            custom_routing_function=partial(
                ernie5_5_routing,
                bias=self.e_score_correction_bias,
                scale=self.gate.routed_scaling_factor_param,
            ),
            router_logits_dtype=torch.float32,
            # fc1 runs outside the runner (see forward()); the norm + fc2 are
            # non-linear, so they run after the expert combine.
            routed_output_transform=Ernie5_5RoutedOutputTransform(
                self.latent_norm, self.fc2_latent_proj
            ),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_experts = self.n_routed_experts
        logits, _ = self.gate(hidden_states)
        scores = logits[:, :num_experts].sigmoid() + logits[:, num_experts:].sigmoid()
        routed_hidden_states, _ = self.fc1_latent_proj(hidden_states)
        return self.experts(
            hidden_states=routed_hidden_states,
            router_logits=scores,
            shared_experts_input=hidden_states,
        )
