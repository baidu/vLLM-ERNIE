# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")

VOCAB = 201216
# FP4 Preview checkpoint values.
RANGES = [15.625, 15.9375, 8.75, 0.3671875]
TS = [-0.1337890625, 0.279296875, -0.09765625, -0.0301513671875]


def _eager(logits, r, t):
    return (
        logits
        + t[0] * (r[0] - logits).relu()
        + t[1] * (logits - r[1]).relu()
        + t[2] * (r[2] - logits).relu().square()
        + t[3] * (logits - r[3]).relu().square()
    )


def _logits(num_tokens, ranges, gen):
    logits = torch.randn(num_tokens, VOCAB, device="cuda", generator=gen) * 12
    r = torch.tensor(ranges, device="cuda")
    edges = torch.cat(
        [
            torch.tensor(
                [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1e-45],
                device="cuda",
            ),
            torch.tensor([-1e-45, 1e-38, 3e38, -3e38, 1e19, -1e19], device="cuda"),
            r,
            torch.nextafter(r, r + 1),
            torch.nextafter(r, r - 1),
            -r,
        ]
    )
    # Edge values at random positions in every row.
    n = num_tokens * 64
    rows = torch.randint(0, num_tokens, (n,), device="cuda", generator=gen)
    cols = torch.randint(0, VOCAB, (n,), device="cuda", generator=gen)
    pick = torch.randint(0, edges.numel(), (n,), device="cuda", generator=gen)
    logits[rows, cols] = edges[pick]
    logits[:, : edges.numel()] = edges
    return logits


def _bits(x):
    return x.view(torch.int32)


def _multimax_worker(rank: int, port: int):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    from vllm_ernie.models.ernie5_5.nvidia.model import Ernie5_5ParallelLMHead

    torch.cuda.set_device(rank)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(1, 0, f"tcp://127.0.0.1:{port}", 0, "nccl")
        initialize_model_parallel(1)
        with torch.device("cuda"):
            head = Ernie5_5ParallelLMHead(VOCAB, 64)
    gen = torch.Generator(device="cuda").manual_seed(0)
    buffers = [(RANGES, TS), (RANGES, [0.0, -1.5, 2.0, 0.0])]
    buffers.append(
        (
            (torch.randn(4, device="cuda", generator=gen) * 8).tolist(),
            torch.randn(4, device="cuda", generator=gen).tolist(),
        )
    )
    for ranges, ts in buffers:
        head.multimax_ranges.copy_(torch.tensor(ranges))
        head.multimax_ts.copy_(torch.tensor(ts))
        r, t = head.multimax_ranges, head.multimax_ts
        for num_tokens in (1, 64, 256, 4096):
            logits = _logits(num_tokens, ranges, gen)
            ref = _eager(logits, r, t)
            out = head.multimax(logits)
            # In place, bitwise equal including NaN payloads and signed zeros.
            assert out.data_ptr() == logits.data_ptr()
            assert torch.equal(_bits(out), _bits(ref)), (ranges, ts, num_tokens)
            del logits, ref, out

    # Rows of a wider buffer: only the viewed columns change.
    wide = _logits(7, RANGES, gen)
    view = wide[:, : VOCAB - 100]
    ref = _eager(view, r, t)
    tail = wide[:, VOCAB - 100 :].clone()
    head.multimax(view)
    assert torch.equal(_bits(view), _bits(ref))
    assert torch.equal(_bits(wide[:, VOCAB - 100 :]), _bits(tail))

    # Non-fp32 logits keep the eager path.
    logits = torch.randn(3, VOCAB, device="cuda").bfloat16()
    ref = _eager(logits, r, t)
    out = head.multimax(logits)
    assert out.dtype == torch.bfloat16 and torch.equal(out, ref)
    assert head.multimax(torch.empty(0, VOCAB, device="cuda")).shape == (0, VOCAB)


def test_multimax_matches_eager():
    import torch.multiprocessing as mp

    from vllm.utils.network_utils import get_open_port

    mp.spawn(_multimax_worker, args=(get_open_port(),), nprocs=1)
