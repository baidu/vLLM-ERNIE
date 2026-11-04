# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="ERNIE 5.5 VHA requires CUDA"
)

NUM_HEADS, RANK, HEAD_DIM = 64, 16, 512


def _inputs(num_tokens: int) -> tuple[torch.Tensor, ...]:
    # CPU: other tests may leave the default device on CUDA.
    gen = torch.Generator().manual_seed(0)
    kw = dict(generator=gen, device="cpu")
    # Checkpoint-like scale.
    u = (torch.randn(NUM_HEADS, RANK, **kw) * 0.3).bfloat16()
    v = (torch.randn(NUM_HEADS, RANK, **kw) * 0.3).bfloat16()
    o = torch.randn(num_tokens, NUM_HEADS, HEAD_DIM, **kw).bfloat16()
    return u, v, o


def _vha_tp_worker(rank: int, world_size: int, port: int, num_tokens: int, out):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    from vllm_ernie.models.ernie5_5.nvidia.attention import VHAPostMix

    torch.cuda.set_device(rank)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size, rank, f"tcp://127.0.0.1:{port}", rank, "nccl"
        )
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        with torch.device("cuda"):
            vha = VHAPostMix(NUM_HEADS).bfloat16()
        u, v, o = (t.cuda() for t in _inputs(num_tokens))
        vha.U.weight_loader(vha.U, u)
        vha.V.weight_loader(vha.V, v)
        vha.fold()
        heads = NUM_HEADS // world_size
        local = o[:, rank * heads : (rank + 1) * heads]
        out[rank] = vha(local).cpu()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("num_tokens", [1, 37])
def test_vha_tp_matches_folded_mix(world_size, num_tokens):
    """Low-rank TP mixing with an all-reduce matches the TP=1 ``m @ o``."""
    from vllm.utils.network_utils import get_open_port

    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    with mp.Manager() as manager:
        out = manager.dict()
        mp.spawn(
            _vha_tp_worker,
            args=(world_size, get_open_port(), num_tokens, out),
            nprocs=world_size,
        )
        tp_out = torch.cat([out[r] for r in range(world_size)], dim=1).float()

    u, v, o = _inputs(num_tokens)
    eye = torch.eye(NUM_HEADS, device="cpu")
    exact = (eye + v.float() @ u.float().t()) @ o.float()
    m = torch.addmm(eye.bfloat16(), v, u.t())
    tp1 = torch.matmul(m.cuda(), o.cuda()).float().cpu()
    # TP>1 mixes in fp32, so it is no less accurate than the bf16 m @ o.
    tp_err, tp1_err = (tp_out - exact).abs(), (tp1 - exact).abs()
    assert tp_err.mean() <= tp1_err.mean()
    assert tp_err.max() <= tp1_err.max()
    torch.testing.assert_close(tp_out, tp1, atol=0.25, rtol=2e-2)


@pytest.mark.parametrize("tp_size", [3, 16])
def test_attn_tp_must_divide_groups_and_heads(monkeypatch, tp_size):
    from vllm_ernie.models.ernie5_5.nvidia import attention

    monkeypatch.setattr(
        attention, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    config = SimpleNamespace(
        o_groups=8, num_attention_heads=64, hybrid_mla_num_attention_heads=64
    )
    with pytest.raises(ValueError, match=f"TP size {tp_size} must divide o_groups"):
        attention._check_attn_tp(config)
