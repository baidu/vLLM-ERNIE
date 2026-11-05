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


@pytest.mark.parametrize("num_rows", [1, 5, 1000])
@pytest.mark.parametrize("hidden", [8192, 16384])
@torch.inference_mode()
def test_sigmoid_mul_quant_matches_unfused(num_rows: int, hidden: int):
    """Fused gate + quant is bitwise the unfused product + DeepGEMM quant."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8_packed_for_deepgemm,
    )

    from vllm_ernie.models.ernie5_5.nvidia.attention import sigmoid_mul_quant_fp8

    torch.manual_seed(0)
    x = torch.randn(num_rows, hidden, device="cuda").bfloat16() * 3
    gate = torch.randn(num_rows, hidden, device="cuda").bfloat16() * 4
    q, s = sigmoid_mul_quant_fp8(x, gate)
    q_ref, s_ref = per_token_group_quant_fp8_packed_for_deepgemm(
        x * torch.sigmoid(gate), 128, use_ue8m0=True
    )
    assert s.shape == s_ref.shape and s.stride() == s_ref.stride()
    assert torch.equal(s, s_ref)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))


def _gated_wo_b_worker(rank: int, port: int, model_dir: str):
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT
    from vllm.utils.torch_utils import set_default_torch_dtype

    import vllm_ernie
    from vllm_ernie.models.ernie5_5.nvidia.attention import GatedDeepGemmLinear

    vllm_ernie.register()
    torch.cuda.set_device(rank)
    model_config = ModelConfig(
        model=model_dir, trust_remote_code=True, skip_tokenizer_init=True
    )
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )
    with set_current_vllm_config(VllmConfig(model_config=model_config)):
        init_distributed_environment(1, 0, f"tcp://127.0.0.1:{port}", 0, "nccl")
        initialize_model_parallel(1)
        with set_default_torch_dtype(torch.bfloat16), torch.device("cuda"):
            wo_b = ReplicatedLinear(
                8192, 4096, bias=False, quant_config=quant_config, prefix="wo_b"
            )
        gen = torch.Generator(device="cuda").manual_seed(0)
        w = torch.randn(4096, 8192, device="cuda", generator=gen) * 0.5
        wo_b.weight.data.copy_(w.to(torch.float8_e4m3fn))
        exps = torch.randint(
            -8, -4, wo_b.weight_scale_inv.shape, device="cuda", generator=gen
        )
        wo_b.weight_scale_inv.data.copy_(torch.exp2(exps.float()))
        wo_b.quant_method.process_weights_after_loading(wo_b)

        fused = GatedDeepGemmLinear.wrap(wo_b)
        assert fused is not None
        x = torch.randn(37, 8192, device="cuda", generator=gen).bfloat16()
        gate = torch.randn(37, 8192, device="cuda", generator=gen).bfloat16() * 4
        assert torch.equal(fused(x, gate), wo_b(x * torch.sigmoid(gate))[0])

        # Hopper: DeepGEMM takes float32 UE8M0 scales, not our packed int32.
        DeepGemmQuantScaleFMT._oracle_cache = DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0
        assert GatedDeepGemmLinear.wrap(wo_b) is None


def test_gated_wo_b_matches_linear(tmp_path):
    """The fused wo_b is bitwise the DeepGEMM linear of ``x * sigmoid(gate)``."""
    import torch.multiprocessing as mp

    from vllm.utils.network_utils import get_open_port

    from vllm_ernie.models.ernie5_5.config import Ernie5_5Config

    config = Ernie5_5Config(num_hidden_layers=1, vocab_size=1024, hidden_size=512)
    config.architectures = ["Ernie5_5_MoeForCausalLM"]
    config.save_pretrained(tmp_path)
    mp.spawn(_gated_wo_b_worker, args=(get_open_port(), str(tmp_path)), nprocs=1)
