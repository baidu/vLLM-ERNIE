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
    from vllm_ernie5_plugin.ernie5.nvidia.quant_config import to_mxfp8_weight_scale

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
    from vllm_ernie5_plugin.ernie5.nvidia.quant_config import Ernie5_5MXFP8Config

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
