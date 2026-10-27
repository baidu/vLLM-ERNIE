# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_torch
from vllm.model_executor.layers.mhc import HAS_TILELANG_MHC
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() and HAS_TILELANG_MHC),
    reason="TileLang MHC support required",
)


def _mhc_pre_ref(residual, fn, hc_scale, hc_base, rms_eps, pre_eps, sk_eps, alpha, n):
    """``mhc_pre_torch`` with the ERNIE RMS factor ``1 / (sqrt(mean) + eps)``."""
    num_tokens, hc_mult, hidden_size = residual.shape
    x = residual.view(num_tokens, -1).float()
    mixes = (x @ fn.t()) / (x.square().mean(dim=-1, keepdim=True).sqrt() + rms_eps)
    pre_mix = torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult])
    post = mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1]
    post_mix = torch.sigmoid(post + hc_base[hc_mult : 2 * hc_mult]) * alpha
    comb = mixes[:, 2 * hc_mult :] * hc_scale[2] + hc_base[2 * hc_mult :]
    comb = torch.softmax(comb.view(-1, hc_mult, hc_mult), dim=-1) + sk_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + sk_eps)
    for _ in range(n - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + sk_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + sk_eps)
    layer_input = ((pre_mix + pre_eps).unsqueeze(-1) * residual.float()).sum(dim=1)
    return post_mix.unsqueeze(-1), comb, layer_input.to(torch.bfloat16)


@pytest.mark.parametrize("num_tokens", [1, 8, 128])
@pytest.mark.parametrize("entry", ["pre", "broadcast", "fused"])
def test_mhc_pre_rms_eps_outside_sqrt(num_tokens, entry):
    """The plugin ops use ``1 / (sqrt(mean) + eps)``; a large eps separates it
    from vLLM's ``rsqrt(mean + eps)``."""
    from vllm_ernie5_plugin.mhc import (
        mhc_fused_post_pre_tilelang,
        mhc_pre_broadcast_tilelang,
        mhc_pre_tilelang,
    )

    torch.set_default_device("cuda")
    set_random_seed(0)
    hidden_size, hc_mult = 4096, 4
    hc_mult3 = 2 * hc_mult + hc_mult * hc_mult
    fn = torch.randn((hc_mult3, hc_mult * hidden_size), dtype=torch.float) * 1e-2
    hc_scale = torch.ones((3,), dtype=torch.float)
    hc_base = torch.randn((hc_mult3,), dtype=torch.float) * 0.1
    weight = torch.randn(hidden_size, dtype=torch.bfloat16)
    rms_eps, eps, repeat, post_alpha = 0.5, 1e-6, 20, 2.0
    args = (fn, hc_scale, hc_base, rms_eps, eps, eps, post_alpha, repeat)
    norm_kwargs = dict(norm_weight=weight, norm_eps=eps)

    x = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16)
    if entry == "broadcast":
        residual = x.unsqueeze(1).expand(-1, hc_mult, -1).contiguous()
        fn_broadcast = fn.view(hc_mult3, hc_mult, hidden_size).sum(dim=1)
        out = mhc_pre_broadcast_tilelang(
            x, *args, **norm_kwargs, fn_broadcast=fn_broadcast
        )[1:]
    elif entry == "pre":
        residual = torch.randn((num_tokens, hc_mult, hidden_size), dtype=torch.bfloat16)
        out = mhc_pre_tilelang(residual, *args, **norm_kwargs)
    else:
        prev = torch.randn((num_tokens, hc_mult, hidden_size), dtype=torch.bfloat16)
        post = torch.rand((num_tokens, hc_mult, 1), dtype=torch.float32)
        comb = torch.rand((num_tokens, hc_mult, hc_mult), dtype=torch.float32)
        residual = mhc_post_torch(x, prev, post, comb)
        res, *out = mhc_fused_post_pre_tilelang(
            x, prev, post, comb, *args, **norm_kwargs
        )
        torch.testing.assert_close(res, residual, atol=1e-2, rtol=1e-2)

    post_ref, comb_ref, layer_input_ref = _mhc_pre_ref(residual, *args)
    layer_input_ref = F.rms_norm(layer_input_ref, (hidden_size,), weight, eps)
    rsqrt_post_ref = mhc_pre_torch(residual, *args)[0]
    assert (rsqrt_post_ref - post_ref).abs().max() > 5e-2
    torch.testing.assert_close(out[0], post_ref, atol=5e-3, rtol=0)
    torch.testing.assert_close(out[1], comb_ref, atol=5e-3, rtol=0)
    torch.testing.assert_close(out[2], layer_input_ref, atol=5e-2, rtol=1e-2)
