# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPLB: redundant experts load logical weights; routing stays logical."""

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

from vllm.config import EPLBConfig, ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.platforms import current_platform

from vllm_ernie.models.ernie5_5.config import Ernie5_5Config
from vllm_ernie.models.ernie5_5.nvidia.model import ernie5_5_expert_mapping

E, R, TOPK = 16, 4, 4
HIDDEN, LATENT, INTER = 64, 32, 64


def _vllm_config(num_redundant: int, world_size: int = 2) -> VllmConfig:
    if not num_redundant:
        return VllmConfig()
    return VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=world_size,
            enable_expert_parallel=True,
            enable_eplb=True,
            eplb_config=EPLBConfig(num_redundant_experts=num_redundant),
        )
    )


@pytest.mark.parametrize("num_redundant", [0, R])
def test_expert_mapping_reads_redundant_experts(num_redundant):
    config = Ernie5_5Config(n_routed_experts=E)
    with set_current_vllm_config(_vllm_config(num_redundant)):
        mapping = ernie5_5_expert_mapping(nn.Module(), config)
    for shard in ("w1", "w2", "w3"):
        entries = {eid: name for _, name, eid, sid in mapping if sid == shard}
        assert sorted(entries) == list(range(E + num_redundant))
        # Physical expert p holds logical expert p % E (initial EPLB layout).
        for p, name in entries.items():
            assert name == f"experts.{p % E}.{shard}."


def _ckpt(gen: torch.Generator) -> dict[str, torch.Tensor]:
    ckpt = {
        "ffn.e_score_correction_bias": torch.randn(E, generator=gen) * 0.1,
        "ffn.gate.routed_scaling_factor_param": torch.rand(E, generator=gen) + 2.5,
    }
    # Logical experts 0..R-1 have replicas; make them win so replicas get routed.
    ckpt["ffn.e_score_correction_bias"][:R] += 1.0
    for e in range(E):
        for proj, shape in (("w1", (INTER, LATENT)), ("w3", (INTER, LATENT))):
            ckpt[f"ffn.experts.{e}.{proj}.weight"] = torch.randn(*shape, generator=gen)
        ckpt[f"ffn.experts.{e}.w2.weight"] = torch.randn(LATENT, INTER, generator=gen)
    return ckpt


def _eplb_worker(rank: int, world_size: int, port: int):
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    from vllm_ernie.models.ernie5_5.nvidia.model import load_ernie5_5_weights
    from vllm_ernie.models.ernie5_5.nvidia.moe import (
        Ernie5_5MoE,
        _ernie5_5_routing_torch,
        split_sigmoid_sum,
    )

    torch.cuda.set_device(rank)
    config = Ernie5_5Config(
        hidden_size=HIDDEN,
        moe_latent_size=LATENT,
        moe_intermediate_size=INTER,
        n_routed_experts=E,
        num_experts_per_tok=TOPK,
        moe_split_feature_routing=True,
        routed_scaling_factor_learnable=True,
    )
    with set_current_vllm_config(_vllm_config(R, world_size)):
        init_distributed_environment(
            world_size, rank, f"tcp://127.0.0.1:{port}", rank, "nccl"
        )
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        model = nn.Module()
        with torch.device("cuda"):
            model.ffn = moe = Ernie5_5MoE(config, prefix="ffn")
        assert moe.n_physical_experts == E + R
        assert moe.n_local_physical_experts == (E + R) // world_size

        ckpt = _ckpt(torch.Generator().manual_seed(0))
        mapping = ernie5_5_expert_mapping(model, config)
        load_ernie5_5_weights(model, ckpt.items(), mapping)

    # Each local physical slot holds its logical expert's checkpoint weights.
    routed = moe.experts.routed_experts
    phys_to_log = EplbState.build_initial_global_physical_to_logical_map(E, R)
    local = 0
    for p, lg in enumerate(phys_to_log):
        j = moe.experts._map_global_expert_id_to_local_expert_id(p)
        if j < 0:
            continue
        local += 1
        w13, w2 = routed.w13_weight[j].cpu(), routed.w2_weight[j].cpu()
        ref13 = torch.cat(
            [ckpt[f"ffn.experts.{lg}.w1.weight"], ckpt[f"ffn.experts.{lg}.w3.weight"]]
        )
        torch.testing.assert_close(w13, ref13.to(w13.dtype))
        ref2 = ckpt[f"ffn.experts.{lg}.w2.weight"]
        torch.testing.assert_close(w2, ref2.to(w2.dtype))
    assert local == moe.n_local_physical_experts

    # Bias and scale stay [E]: routing picks logical ids, then vLLM maps them
    # to physical replicas.
    bias, scale = moe.e_score_correction_bias, moe.gate.routed_scaling_factor_param
    assert bias.shape == scale.shape == (E,)
    phys_to_log = torch.tensor(phys_to_log, device="cuda")
    count = torch.bincount(phys_to_log, minlength=E)
    log_to_phys = torch.full((E, int(count.max())), -1, device="cuda")
    for lg in range(E):
        replicas = (phys_to_log == lg).nonzero().flatten()
        log_to_phys[lg, : replicas.numel()] = replicas
    load = torch.zeros(1, E + R, dtype=torch.int32, device="cuda")
    moe.experts.set_eplb_state(0, load, log_to_phys[None], count[None])
    T = 64
    eplb = moe.experts.eplb_state
    eplb.should_record_tensor = torch.ones(1, dtype=torch.int32, device="cuda")
    eplb.num_unpadded_tokens_tensors = [torch.tensor([T], device="cuda")]

    gen = torch.Generator(device="cuda").manual_seed(1)
    logits = torch.randn(T, 2 * E, device="cuda", generator=gen) * 4
    scores = split_sigmoid_sum(logits, E)
    weights, phys = moe.experts.router.select_experts(
        torch.zeros(T, LATENT, device="cuda"), scores
    )
    ref_w, ref_ids = _ernie5_5_routing_torch(scores, TOPK, True, bias, scale)
    ids = phys_to_log[phys.long()]
    order, ref_order = ids.sort(-1), ref_ids.sort(-1)
    torch.testing.assert_close(order.values.int(), ref_order.values)
    torch.testing.assert_close(
        weights.gather(1, order.indices), ref_w.gather(1, ref_order.indices)
    )
    assert (phys >= E).any(), "no token was routed to a redundant replica"
    assert int(load.sum()) == T * TOPK


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
def test_eplb_loads_replicas_and_routes_logical_ids():
    """2-rank EP with redundant experts on a real `Ernie5_5MoE`."""
    from vllm.utils.network_utils import get_open_port

    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    mp.spawn(_eplb_worker, args=(2, get_open_port()), nprocs=2)
