# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")


@pytest.mark.parametrize("num_tokens", [1, 7, 4096])
def test_indexer_partial_rope_in_place(num_tokens):
    """Partial RoPE on the full head equals RoPE on the split rope part + cat."""
    set_random_seed(0)
    n_head, head_dim, rope_dim = 64, 128, 64
    params = {"rope_type": "default", "rope_theta": 8000000}
    with set_current_vllm_config(VllmConfig()):
        rope = get_rope(
            rope_dim, max_position=65536, is_neox_style=True, rope_parameters=params
        ).cuda()
        # As built by Ernie5_5Indexer.
        partial = get_rope(
            head_dim,
            max_position=65536,
            is_neox_style=True,
            rope_parameters={**params, "partial_rotary_factor": rope_dim / head_dim},
        ).cuda()
    assert partial.rotary_dim == rope_dim
    pos = torch.randint(0, 65536, (num_tokens,), device="cuda")
    q = torch.randn(num_tokens, n_head, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(num_tokens, head_dim, device="cuda", dtype=torch.bfloat16)

    nope_dim = head_dim - rope_dim
    q_pe, q_nope = q.clone().split([rope_dim, nope_dim], dim=-1)
    k_pe, k_nope = k.clone().split([rope_dim, nope_dim], dim=-1)
    # The served model runs the CUDA kernel (custom ops are on).
    q_pe, k_pe = rope.forward_cuda(pos, q_pe, k_pe.unsqueeze(1))
    ref_q = torch.cat([q_pe.reshape(-1, n_head, rope_dim), q_nope], -1)
    ref_k = torch.cat([k_pe.reshape(-1, rope_dim), k_nope], dim=-1)

    out_q, out_k = partial.forward_cuda(pos, q, k)
    # No copy: the kernel rotates q and k in place.
    assert out_q.data_ptr() == q.data_ptr() and out_k.data_ptr() == k.data_ptr()
    assert torch.equal(out_q, ref_q)
    assert torch.equal(out_k, ref_k)
