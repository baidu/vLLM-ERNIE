# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark draft multimax after the Markov bias, before draft sampling."""

import pytest
import torch
from torch import nn

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

from vllm_ernie.models.ernie5_5.patches import patch_dspark_sample_logits

VOCAB = 201216
# Checkpoint values, then positive t[0] and t[2], which turn -inf into NaN.
RANGES = [15.625, 15.9375, 8.75, 0.3671875]
TS = (
    [-0.1337890625, 0.279296875, -0.09765625, -0.0301513671875],
    [0.25, 0.279296875, 0.0625, -0.0301513671875],
)
NEG_INF = float("-inf")


def _eager(logits, r, t):
    return (
        logits
        + t[0] * (r[0] - logits).relu()
        + t[1] * (logits - r[1]).relu()
        + t[2] * (r[2] - logits).relu().square()
        + t[3] * (logits - r[3]).relu().square()
    )


def _bits(x):
    return x.view(torch.int32)


def _worker(rank: int, port: int):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    from vllm_ernie.models.ernie5_5.nvidia.dspark import Ernie5_5DSparkForCausalLM
    from vllm_ernie.models.ernie5_5.nvidia.model import Ernie5_5ParallelLMHead

    torch.cuda.set_device(rank)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(1, 0, f"tcp://127.0.0.1:{port}", 0, "nccl")
        initialize_model_parallel(1)
        with torch.device("cuda"):
            head = Ernie5_5ParallelLMHead(VOCAB, 64)
    draft = Ernie5_5DSparkForCausalLM.__new__(Ernie5_5DSparkForCausalLM)
    nn.Module.__init__(draft)
    draft.lm_head = head
    gen = torch.Generator(device="cuda").manual_seed(0)

    for ts in TS:
        head.multimax_ranges.copy_(torch.tensor(RANGES))
        head.multimax_ts.copy_(torch.tensor(ts))
        r, t = head.multimax_ranges, head.multimax_ts

        # Finite logits match multimax bitwise.
        for num_tokens in (1, 7, 128 * 7):
            logits = torch.randn(num_tokens, VOCAB, device="cuda", generator=gen) * 12
            ref = _eager(logits, r, t)
            out = draft.postprocess_draft_logits(logits)
            assert torch.equal(_bits(out), _bits(ref)), (ts, num_tokens)

        # Top-k path: a strided row of [reqs, steps, vocab], -inf outside the
        # candidates, plus NaN entries.
        base = torch.full((5, 7, VOCAB), NEG_INF, device="cuda")
        cols = torch.randint(0, VOCAB, (5, 7, 512), device="cuda", generator=gen)
        vals = torch.randn(5, 7, 512, device="cuda", generator=gen) * 12
        base.scatter_(2, cols, vals)
        base[:, :, 7] = float("nan")
        other = base[:, 3].clone()
        for logits in (base[:, 2], base[:, 2].bfloat16()):
            before = logits.clone()
            ref = _eager(before, r, t)
            out = draft.postprocess_draft_logits(logits)
            masked = before.isneginf()
            finite = before.isfinite()
            assert out.dtype == before.dtype
            assert out[masked].isneginf().all()
            assert out[:, 7].isnan().all()
            assert torch.equal(out[finite], ref[finite])
            if ts is TS[1]:
                assert ref[masked].isnan().all()
        assert torch.equal(_bits(base[:, 3]), _bits(other))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
def test_postprocess_draft_logits_matches_multimax():
    import torch.multiprocessing as mp

    from vllm.utils.network_utils import get_open_port

    mp.spawn(_worker, args=(get_open_port(),), nprocs=1)


class _Draft:
    def __init__(self):
        self.calls = []

    def postprocess_draft_logits(self, logits):
        self.calls.append(logits)
        out = logits.clone()
        out[:, 5] = 1.0
        return out

    def map_draft_to_target(self, draft_ids):
        return draft_ids


def test_sample_logits_calls_draft_hook_once():
    patch_dspark_sample_logits()
    wrapped = DSparkSpeculator._sample_logits
    patch_dspark_sample_logits()
    assert DSparkSpeculator._sample_logits is wrapped
    assert not getattr(wrapped.__wrapped__, "_ernie5_draft_logits", False)

    spec = DSparkSpeculator.__new__(DSparkSpeculator)
    spec.acceptance_estimator = None
    spec.draft_logits = None  # greedy drafting
    spec._step_cols = torch.arange(3, dtype=torch.int32)
    spec.model = _Draft()
    logits = torch.zeros(3, 10)
    idx, pos = torch.arange(3), torch.ones(3, dtype=torch.int64)
    for step in range(3):
        assert spec._sample_logits(logits, idx, pos, step).tolist() == [5, 5, 5]
        assert len(spec.model.calls) == step + 1
        assert spec.model.calls[-1] is logits
