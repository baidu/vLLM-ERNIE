# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="ERNIE 5.5 requires CUDA"
)


def test_mxfp8_scale_is_lossless():
    """32x32 power-of-two block scales dequantize identically as MXFP8."""
    from vllm_ernie.models.ernie5_5.nvidia.quant_config import to_mxfp8_weight_scale

    n, k = 64, 96
    exp = torch.randint(-12, 4, (n // 32, k // 32))
    scale = torch.exp2(exp.float())
    q = torch.randn(n, k).clamp(-448, 448).to(torch.float8_e4m3fn)
    ref = q.float() * scale.repeat_interleave(32, 0).repeat_interleave(32, 1)

    mx_scale = to_mxfp8_weight_scale(scale)
    assert mx_scale.dtype == torch.uint8 and mx_scale.shape == (n, k // 32)
    torch.testing.assert_close(
        dequant_mxfp8_to_bf16(q, mx_scale), ref.bfloat16(), rtol=0, atol=0
    )

    with pytest.raises(ValueError):
        to_mxfp8_weight_scale(scale * 1.5)


@pytest.mark.parametrize(
    ("block", "scale_fmt", "expected"),
    [([32, 32], "ue8m0", True), ([32, 32], None, False), ([128, 128], "ue8m0", False)],
)
def test_mxfp8_only_for_b32_ue8m0(block, scale_fmt, expected):
    """Only 32x32 ue8m0 checkpoints switch to MXFP8, and only on Blackwell."""
    from vllm_ernie.models.ernie5_5.nvidia.quant_config import Ernie5_5MXFP8Config

    hf_cfg = {"quant_method": "fp8", "activation_scheme": "dynamic"}
    hf_cfg["weight_block_size"] = block
    if scale_fmt:
        hf_cfg["scale_fmt"] = scale_fmt
    base = Fp8Config.from_config(hf_cfg)
    wrapped = Ernie5_5MXFP8Config.maybe_wrap(base, hf_cfg)
    blackwell = current_platform.is_device_capability_family(100)
    assert isinstance(wrapped, Ernie5_5MXFP8Config) == (expected and blackwell)
    if wrapped is not base:
        assert wrapped.ignored_layers == base.ignored_layers


def _fp8_config(ignored=None):
    return Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        ignored_layers=ignored,
        weight_block_size=[128, 128],
    )


@pytest.mark.parametrize(
    ("ignored", "expected"),
    [
        # Release 128x128 checkpoint: no ignore list, wo_a is FP8.
        (None, True),
        (["model.layers.0.self_attn.linear_o_group_proj"], False),
        (["model.layers.0.attn.wo_a"], False),
    ],
)
def test_keeps_fp8_wo_a(ignored, expected):
    from vllm_ernie.models.ernie5_5.nvidia.quant_config import keeps_fp8_wo_a

    assert keeps_fp8_wo_a(_fp8_config(ignored)) is expected
    assert not keeps_fp8_wo_a(None)


def test_dequantize_block_fp8_is_exact():
    from vllm_ernie.models.ernie5_5.nvidia.quant_config import dequantize_block_fp8

    n, k = 160, 4096
    scale = torch.exp2(torch.randint(-12, 4, (-(-n // 128), k // 128)).float())
    q = torch.randn(n, k).clamp(-448, 448).to(torch.float8_e4m3fn)
    ref = q.float() * scale.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)
    out = dequantize_block_fp8(q, scale, [128, 128])
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref, rtol=0, atol=0)


def test_loader_dequantizes_fp8_into_bf16_params():
    """FP8 weights of bf16 params are dequantized; FP8 params pass through."""
    from torch import nn

    from vllm_ernie.models.ernie5_5.nvidia.model import _dequantize_bf16_layers

    params = {
        "layers.6.attn.indexer.wk_weights_proj.weight": nn.Parameter(
            torch.empty(192, 256, dtype=torch.bfloat16), requires_grad=False
        ),
        "layers.0.attn.wq_b.weight": nn.Parameter(
            torch.empty(256, 256, dtype=torch.float8_e4m3fn), requires_grad=False
        ),
        "layers.0.attn.wq_b.weight_scale_inv": nn.Parameter(
            torch.empty(2, 2), requires_grad=False
        ),
    }
    q = torch.randn(128, 256).to(torch.float8_e4m3fn)
    s = torch.full((1, 2), 0.5)
    wk, wk_s = q.clone(), s.clone()
    wk.shard_id = wk_s.shard_id = 0
    weights = [
        ("layers.6.attn.indexer.wk_weights_proj.weight_scale_inv", wk_s),
        ("layers.0.attn.wq_b.weight", torch.zeros(256, 256).to(torch.float8_e4m3fn)),
        ("layers.6.attn.indexer.wk_weights_proj.weight", wk),
        ("layers.0.attn.wq_b.weight_scale_inv", torch.ones(2, 2)),
    ]
    pending = {}
    out = list(_dequantize_bf16_layers(weights, params, pending, [128, 128]))
    assert not pending
    names = [n for n, _ in out]
    assert names == [
        "layers.0.attn.wq_b.weight",
        "layers.6.attn.indexer.wk_weights_proj.weight",
        "layers.0.attn.wq_b.weight_scale_inv",
    ]
    wk_out = out[1][1]
    assert wk_out.dtype == torch.bfloat16 and wk_out.shard_id == 0
    torch.testing.assert_close(wk_out.float(), q.float() * 0.5, rtol=0, atol=0)
