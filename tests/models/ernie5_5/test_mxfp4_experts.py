# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 routed experts declared by ``expert_dtype: mxfp4`` in the config."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="ERNIE 5.5 requires CUDA"
)

HF_QUANT = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "scale_fmt": "ue8m0",
    "weight_block_size": [128, 128],
}


@pytest.mark.parametrize(
    ("top_level", "in_quant_cfg", "expected"),
    [
        ("mxfp4", None, True),
        (None, "mxfp4", True),
        (None, None, False),
        ("fp8", None, False),
    ],
)
def test_mxfp4_experts_read_from_config(top_level, in_quant_cfg, expected):
    from vllm_ernie.models.ernie5_5.nvidia.quant_config import (
        use_ernie5_5_mxfp4_experts,
    )

    hf_quant_cfg = dict(HF_QUANT)
    if in_quant_cfg is not None:
        hf_quant_cfg["expert_dtype"] = in_quant_cfg
    hf_config = SimpleNamespace(quantization_config=hf_quant_cfg)
    if top_level is not None:
        hf_config.expert_dtype = top_level
    quant_config = Fp8Config.from_config(hf_quant_cfg)
    vllm_config = SimpleNamespace(
        quant_config=quant_config, model_config=SimpleNamespace(hf_config=hf_config)
    )
    assert use_ernie5_5_mxfp4_experts(vllm_config) is expected
    # vLLM picks `Mxfp4MoEMethod` from `store_dtype`.
    assert quant_config.store_dtype == ("mxfp4" if expected else None)
    assert not use_ernie5_5_mxfp4_experts(SimpleNamespace(quant_config=None))


def test_unknown_expert_dtype_rejected():
    from vllm_ernie.models.ernie5_5.nvidia.quant_config import (
        use_ernie5_5_mxfp4_experts,
    )

    hf_config = SimpleNamespace(quantization_config=HF_QUANT, expert_dtype="fp4")
    vllm_config = SimpleNamespace(
        quant_config=Fp8Config.from_config(HF_QUANT),
        model_config=SimpleNamespace(hf_config=hf_config),
    )
    with pytest.raises(ValueError, match="expert_dtype"):
        use_ernie5_5_mxfp4_experts(vllm_config)


class _Experts(nn.Module):
    """`Mxfp4MoEMethod` parameter names, with a recording weight loader."""

    def __init__(self, num_experts=2, n=64, k=128):
        super().__init__()
        self.calls = []
        shapes = {
            "w13_weight": (num_experts, 2 * n, k // 2),
            "w13_weight_scale": (num_experts, 2 * n, k // 32),
            "w2_weight": (num_experts, k, n // 2),
            "w2_weight_scale": (num_experts, k, n // 32),
        }
        for name, shape in shapes.items():
            p = nn.Parameter(torch.zeros(shape, dtype=torch.uint8), requires_grad=False)
            p.weight_loader = self._loader
            self.register_parameter(name, p)

    def _loader(self, param, w, name, shard_id, expert_id, return_success=False):
        self.calls.append((name, shard_id, expert_id, w.dtype, tuple(w.shape)))
        return True


def _model():
    model = nn.Module()
    model.layers = nn.ModuleDict({"1": nn.Module()})
    model.layers["1"].ffn = nn.Module()
    model.layers["1"].ffn.experts = nn.Module()
    model.layers["1"].ffn.experts.routed_experts = _Experts()
    model.mxfp4_experts = True
    return model


def test_mxfp4_expert_weight_names(monkeypatch):
    """`weight_scale_inv` loads into `weight_scale`; int8 weights as uint8."""
    from vllm_ernie.models.ernie5_5.nvidia import model as model_mod

    monkeypatch.setattr(model_mod, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(model_mod, "get_tensor_model_parallel_rank", lambda: 0)
    model = _model()
    mapping = model_mod.ernie5_5_expert_mapping(
        model, SimpleNamespace(n_routed_experts=2)
    )
    n, k = 64, 128
    weights = []
    for e in range(2):
        for proj, shape in (("w1", (n, k)), ("w3", (n, k)), ("w2", (k, n))):
            base = f"layers.1.ffn.experts.{e}.{proj}"
            packed = torch.randint(
                -128, 128, (shape[0], shape[1] // 2), dtype=torch.int8
            )
            scale = torch.randint(
                100, 140, (shape[0], shape[1] // 32), dtype=torch.uint8
            )
            weights.append((f"{base}.weight", packed))
            weights.append(
                (f"{base}.weight_scale_inv", scale.view(torch.float8_e8m0fnu))
            )
    loaded = model_mod.load_ernie5_5_weights(model, weights, mapping)

    prefix = "layers.1.ffn.experts.routed_experts."
    assert loaded == {
        prefix + p
        for p in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    }
    calls = model.layers["1"].ffn.experts.routed_experts.calls
    assert len(calls) == 12
    for name, shard_id, _, dtype, shape in calls:
        assert not name.endswith("_inv")
        if name.endswith("_weight"):
            assert dtype == torch.uint8
        else:
            assert dtype == torch.float8_e8m0fnu
        assert shape[1] == (n if shard_id == "w2" else k) // (
            2 if name.endswith("_weight") else 32
        )
    assert {(c[1], c[2]) for c in calls} == {
        (s, e) for s in ("w1", "w2", "w3") for e in (0, 1)
    }


def test_bundled_mtp_weights_skipped(monkeypatch):
    """`mtp.*` (bundled DSpark draft) never reaches the target loader."""
    from vllm_ernie.models.ernie5_5.nvidia import model as model_mod

    seen = []
    monkeypatch.setattr(
        model_mod,
        "load_ernie5_5_weights",
        lambda model, weights, mapping: seen.extend(n for n, _ in weights) or set(),
    )
    fake = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=2), get_expert_mapping=lambda: []
    )
    names = [
        "layers.0.a",
        "layers.1.b",
        "layers.2.c",
        "mtp.0.x",
        "mtp.feature_norm.weight",
        "norm.weight",
    ]
    model_mod.Ernie5_5Model.load_weights(fake, ((n, torch.zeros(1)) for n in names))
    assert seen == ["layers.0.a", "layers.1.b", "norm.weight"]
