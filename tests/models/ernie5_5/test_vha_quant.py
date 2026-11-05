# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused VHA + inverse RoPE + FP8 quant vs the unfused HCA output path."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="Packed UE8M0 scales need Blackwell",
)


def _unpack(scale: torch.Tensor) -> torch.Tensor:
    """``[T, G, P]`` packed int32 exponents -> ``[T, G, 4 * P]`` fp32 scales."""
    shifts = torch.arange(4, device=scale.device, dtype=torch.int32) * 8
    exps = (scale.unsqueeze(-1) >> shifts) & 0xFF
    return torch.exp2(exps.flatten(-2).float() - 127)


def _dequant(data: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    t, g, d = data.shape
    s = _unpack(scale).repeat_interleave(128, -1)
    return data.float() * s[..., :d]


@pytest.mark.parametrize("num_tokens", [1, 7, 300])
@torch.inference_mode()
def test_vha_inv_rope_fp8_quant_matches_unfused(num_tokens: int):
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
        fused_inv_rope_fp8_quant,
    )

    from vllm_ernie.models.ernie5_5.nvidia.attention import vha_inv_rope_fp8_quant

    torch.manual_seed(0)
    heads, groups, head_dim, rope_dim, rank = 64, 8, 512, 64, 16
    nope_dim = head_dim - rope_dim
    # Padded head axis like the FlashMLA output buffer.
    o_padded = torch.randn(num_tokens, 128, head_dim, device="cuda").bfloat16()
    o = o_padded[:, :heads]
    u = torch.randn(heads, rank, device="cuda") / 8
    v = torch.randn(heads, rank, device="cuda") / 8
    mix = (torch.eye(heads, device="cuda") + v @ u.T).bfloat16()
    positions = torch.randint(0, 4096, (num_tokens,), device="cuda")
    inv = 1.0 / (10000 ** (torch.arange(0, rope_dim, 2, device="cuda") / rope_dim))
    freqs = torch.arange(4096, device="cuda")[:, None].float() * inv
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], -1)

    ref_data, ref_scale = fused_inv_rope_fp8_quant(
        torch.matmul(mix, o),
        positions,
        cos_sin,
        n_groups=groups,
        heads_per_group=heads // groups,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        tma_aligned_scales=True,
    )
    out = vha_inv_rope_fp8_quant(o, mix, positions, cos_sin, groups, nope_dim, rope_dim)
    assert out.data.shape == ref_data.shape and out.data.stride() == ref_data.stride()
    assert out.scale.shape == ref_scale.shape
    assert out.scale.stride() == ref_scale.stride()
    # Only the bf16 rounding of the mixed output differs (one fp8 step or a
    # scale exponent at a power-of-two boundary).
    assert (out.scale != ref_scale).float().mean() < 0.05
    got, ref = _dequant(out.data, out.scale), _dequant(ref_data, ref_scale)
    exact = torch.matmul(mix.float(), o.float())
    assert ((got - ref).norm() / ref.norm()).item() < 0.03
    # Fused is at least as close to the fp32 mix as the unfused path.
    exact_ref = fused_inv_rope_fp8_quant(
        exact.bfloat16(), positions, cos_sin, groups, heads // groups,
        nope_dim, rope_dim, tma_aligned_scales=True, quantize=False,
    )[0].float()  # fmt: skip
    err = ((got - exact_ref).norm() / exact_ref.norm()).item()
    err_ref = ((ref - exact_ref).norm() / exact_ref.norm()).item()
    assert err <= err_ref * 1.05, (err, err_ref)
