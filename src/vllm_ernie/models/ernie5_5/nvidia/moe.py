# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense FFN and latent MoE blocks for ERNIE 5 (NVIDIA)."""

from functools import partial

import torch
from torch import nn

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.activation import SituAndMul
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.triton_utils import tl, tldevice, triton
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel

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


@triton.jit
def _sigmoid(x):
    # torch's fp32 CUDA sigmoid: 1 / (1 + expf(-x)), IEEE division.
    return tldevice.div_rn(1.0, 1.0 + tldevice.exp(-x))


@triton.jit
def _split_sigmoid_sum_kernel(logits_ptr, out_ptr, stride, E: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, E)
    l0 = tl.load(logits_ptr + row * stride + offs)
    l1 = tl.load(logits_ptr + row * stride + E + offs)
    tl.store(out_ptr + row * E + offs, _sigmoid(l0) + _sigmoid(l1))


def split_sigmoid_sum(logits: torch.Tensor, num_experts: int) -> torch.Tensor:
    """``sigmoid(logits[:, :E]) + sigmoid(logits[:, E:])`` in one kernel."""
    num_tokens = logits.shape[0]
    out = torch.empty(
        (num_tokens, num_experts), dtype=torch.float32, device=logits.device
    )
    if (
        num_tokens
        and logits.stride(1) == 1
        and triton.next_power_of_2(num_experts) == num_experts
    ):
        _split_sigmoid_sum_kernel[(num_tokens,)](
            logits,
            out,
            logits.stride(0),
            E=num_experts,
            # Keep denormal results, as torch does.
            enable_reflect_ftz=False,
        )
    elif num_tokens:
        torch.add(
            logits[:, :num_experts].sigmoid(),
            logits[:, num_experts:].sigmoid(),
            out=out,
        )
    return out


@triton.jit
def _ernie5_5_routing_kernel(
    scores_ptr,
    bias_ptr,
    scale_ptr,
    pad_ptr,
    weights_ptr,
    ids_ptr,
    stride,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_PAD: tl.constexpr,
    RENORM: tl.constexpr,
    HAS_PAD: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, E)
    kofs = tl.arange(0, TOPK_PAD)
    scores = tl.load(scores_ptr + row * stride + offs)
    sel = scores + tl.load(bias_ptr + offs)
    ids = tl.full((TOPK_PAD,), E, dtype=tl.int32)
    vals = tl.zeros((TOPK_PAD,), dtype=tl.float32)
    weights = tl.zeros((TOPK_PAD,), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        best = tl.max(sel, 0)
        idx = tl.minimum(tl.min(tl.where(sel == best, offs, E), 0), E - 1)
        hit = offs == idx
        ids = tl.where(kofs == k, idx, ids)
        vals = tl.where(kofs == k, best, vals)
        weights = tl.where(kofs == k, tl.sum(tl.where(hit, scores, 0.0), 0), weights)
        sel = tl.where(hit, float("-inf"), sel)
    # Reorder to torch.topk(sorted=False) slots: experts above the k-th value
    # by index, then those equal to it by index.
    kmask = kofs < TOPK
    above = kmask & (vals > best)
    key = tl.where(above, ids, ids + E)
    slot = tl.sum((key[None, :] < key[:, None]).to(tl.int32), 1)
    move = slot[None, :] == kofs[:, None]
    ids = tl.sum(tl.where(move, ids[None, :], 0), 1)
    weights = tl.sum(tl.where(move, weights[None, :], 0.0), 1)
    if RENORM:
        # tl.sum's tree over the zero-padded slots adds in the same order as
        # torch's sum(-1); `/` would be approximate, torch divides with IEEE.
        total = tl.maximum(tl.sum(weights, 0), 1e-20)
        weights = tldevice.div_rn(weights, total)
    weights = weights * tl.load(scale_ptr + ids, mask=kmask, other=0.0)
    if HAS_PAD:
        # Padding rows select no expert.
        is_pad = tl.load(pad_ptr + row) != 0
        weights = tl.where(is_pad, 0.0, weights)
        ids = tl.where(is_pad, -1, ids)
    tl.store(weights_ptr + row * TOPK + kofs, weights, mask=kmask)
    tl.store(ids_ptr + row * TOPK + kofs, ids, mask=kmask)


def _ernie5_5_routing_torch(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    bias: torch.Tensor,
    scale: torch.Tensor,
    is_padding: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.topk(gating_output + bias, topk, dim=-1, sorted=False).indices
    weights = gating_output.gather(1, ids)
    if renormalize:
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-20)
    weights = weights * scale[ids]
    weights, ids = weights.float(), ids.int()
    if is_padding is not None:
        is_padding = is_padding.unsqueeze(1)
        weights = weights.masked_fill(is_padding, 0)
        ids = ids.masked_fill(is_padding, -1)
    return weights, ids


def _padding_mask(num_tokens: int) -> torch.Tensor | None:
    # Same mask as vLLM's built-in routers (VLLM_MOE_SKIP_PADDING).
    if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
        is_padding = get_forward_context().is_padding
        # Naive DP dispatch routes the gathered batch, longer than the mask.
        if is_padding is not None and is_padding.numel() >= num_tokens:
            return is_padding[:num_tokens]
    return None


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
    indexed by logical expert id. CUDA graph padding rows get id -1 and
    weight 0. Runs as one Triton kernel when it can.
    """
    num_tokens, num_experts = gating_output.shape
    is_padding = _padding_mask(num_tokens)
    if (
        gating_output.dtype != torch.float32
        or gating_output.stride(1) != 1
        or triton.next_power_of_2(num_experts) != num_experts
    ):
        return _ernie5_5_routing_torch(
            gating_output, topk, renormalize, bias, scale, is_padding
        )
    weights = torch.empty(
        (num_tokens, topk), dtype=torch.float32, device=gating_output.device
    )
    ids = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=gating_output.device
    )
    if num_tokens:
        _ernie5_5_routing_kernel[(num_tokens,)](
            gating_output,
            bias,
            scale,
            is_padding if is_padding is not None else ids,
            weights,
            ids,
            gating_output.stride(0),
            E=num_experts,
            TOPK=topk,
            TOPK_PAD=triton.next_power_of_2(topk),
            RENORM=renormalize,
            HAS_PAD=is_padding is not None,
            enable_reflect_ftz=False,
            # One warp per row, about 2x faster at 8K tokens than 4 warps.
            num_warps=1,
        )
    return weights, ids


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
        aux_stream: torch.cuda.Stream | None = None,
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

        parallel_config = get_current_vllm_config().parallel_config
        # EPLB remaps ids after routing; bias and scale stay indexed by logical id.
        self.enable_eplb = parallel_config.enable_eplb

        hidden_size = config.hidden_size
        latent_size = config.moe_latent_size
        num_experts = config.n_routed_experts
        self.n_routed_experts = num_experts
        self.n_shared_experts = config.n_shared_experts
        self.ep_size = get_ep_group().world_size
        self.ep_rank = get_ep_group().rank_in_group
        self.n_logical_experts = num_experts
        self.n_redundant_experts = parallel_config.eplb_config.num_redundant_experts
        self.n_physical_experts = num_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
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
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            # fc1 runs outside the runner (see forward()); the norm + fc2 are
            # non-linear, so they run after the expert combine.
            routed_output_transform=Ernie5_5RoutedOutputTransform(
                self.latent_norm, self.fc2_latent_proj
            ),
        )
        # fc1 runs on this stream next to the router GEMM for small batches.
        self.aux_stream = aux_stream
        self.events = [torch.cuda.Event(), torch.cuda.Event()]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        def route() -> torch.Tensor:
            logits, _ = self.gate(hidden_states)
            return split_sigmoid_sum(logits, self.n_routed_experts)

        aux_stream = self.aux_stream
        if hidden_states.shape[0] > envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD:
            aux_stream = None
        scores, routed_hidden_states = maybe_execute_in_parallel(
            route,
            lambda: self.fc1_latent_proj(hidden_states)[0],
            self.events[0],
            self.events[1],
            aux_stream,
        )
        return self.experts(
            hidden_states=routed_hidden_states,
            router_logits=scores,
            shared_experts_input=hidden_states,
        )
