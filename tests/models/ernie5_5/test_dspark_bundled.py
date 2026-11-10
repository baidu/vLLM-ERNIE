# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark draft bundled in the target checkpoint under ``model.mtp.*``."""

import json
import os
import re
from types import SimpleNamespace

import pytest
import torch

from vllm_ernie.models.ernie5_5.config import Ernie5_5Config
from vllm_ernie.models.ernie5_5.patches import _default_bundled_dspark_tokens

# Optional real checkpoints: a bundling target and a standalone draft export.
BUNDLED_CKPT = os.environ.get("ERNIE5_BUNDLED_DSPARK_CKPT")
STANDALONE_CKPT = os.environ.get("ERNIE5_STANDALONE_DSPARK_CKPT")


def _bundling_config(**kwargs) -> Ernie5_5Config:
    return Ernie5_5Config(
        num_hidden_layers=4,
        num_nextn_predict_layers=0,
        compress_ratios=[128, -2, 128, 4],
        dspark_target_layer_ids=[1, 2, 3],
        dspark_noise_token_id=1024,
        dspark_markov_rank=256,
        dspark_block_size=7,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("method", "model", "tokens", "expected"),
    [
        ("dspark", None, None, 7),
        ("dspark", None, 3, 3),
        ("dspark", "/some/draft", None, None),
        ("mtp", None, None, None),
    ],
)
def test_bundled_dspark_tokens_default_to_block_size(method, model, tokens, expected):
    spec = SimpleNamespace(
        method=method,
        model=model,
        num_speculative_tokens=tokens,
        target_model_config=SimpleNamespace(hf_config=_bundling_config()),
    )
    _default_bundled_dspark_tokens(spec)
    assert spec.num_speculative_tokens == expected


def _index_names(ckpt: str) -> list[str]:
    with open(os.path.join(ckpt, "model.safetensors.index.json")) as f:
        return list(json.load(f)["weight_map"])


def _map_names(names: list[str], bundled: bool) -> list[str]:
    from vllm_ernie.models.ernie5_5.nvidia.dspark import (
        Ernie5_5DSparkForCausalLM,
        _unbundle_dspark_weights,
    )

    fake = SimpleNamespace(model=SimpleNamespace(confidence_head=object()))
    weights = ((n, torch.empty(1, 1)) for n in names)
    if bundled:
        weights = _unbundle_dspark_weights(weights)
    return [n for n, _ in Ernie5_5DSparkForCausalLM._map_weights(fake, weights)]


def _shape(name: str) -> str:
    # One entry per expert projection; FP8 scales are ckpt-format only.
    return re.sub(r"\.experts\.\d+\.", ".experts.E.", name.replace("_scale_inv", ""))


@pytest.mark.skipif(
    not (BUNDLED_CKPT and STANDALONE_CKPT), reason="needs real checkpoints"
)
def test_bundled_names_match_standalone_export():
    """Every `model.mtp.*` tensor maps like the standalone export's tensor."""
    bundled = [n for n in _index_names(BUNDLED_CKPT) if n.startswith("model.mtp.")]
    standalone = [
        n
        for n in _index_names(STANDALONE_CKPT)
        if not n.startswith("model.lm_head.")  # target-owned multimax
    ]
    mapped_bundled = _map_names(bundled, bundled=True)
    mapped_standalone = _map_names(standalone, bundled=False)
    # Each bundled tensor yields a weight (linear_kv_proj yields two).
    kv = sum(".self_attn.linear_kv_proj." in n for n in bundled)
    assert len(mapped_bundled) == len(bundled) + kv
    assert {_shape(n) for n in mapped_bundled} == {_shape(n) for n in mapped_standalone}
    # The FP8 linears bring their scales, including the shared projections.
    for name in (
        "model.main_proj.weight_scale_inv",
        "model.context_wkv_proj.weight_scale_inv",
    ):
        assert name in mapped_bundled, name
