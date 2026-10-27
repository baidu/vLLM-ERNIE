# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The plugin hooks reproduce the eblite-v1 changes to shared vLLM code."""

import pickle
from types import SimpleNamespace
from typing import get_args

import vllm.config.speculative as spec_mod
from vllm.config import SpeculativeConfig
from vllm.config.vllm import default_breakable_cudagraph_architectures
from vllm.model_executor.models.registry import ModelRegistry
from vllm.transformers_utils.config import _CONFIG_REGISTRY
from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
)

from vllm_ernie.models import ernie5_5
from vllm_ernie.models.ernie5_5.config import Ernie5_5Config
from vllm_ernie.models.ernie5_5.patches import _undo_dspark_rewrite


def test_ernie5_5_dsa_layers_use_hybrid_mla_dims():
    """DSA layers (``compress_ratios <= 0``) must not read the HCA dims."""
    config = Ernie5_5Config(
        num_hidden_layers=3,
        compress_ratios=[128, -2, 128, -2],
        hybrid_mla_qk_nope_head_dim=192,
        hybrid_mla_v_head_dim=256,
    )

    assert [config.is_dsa_layer(i) for i in range(4)] == [False, True, False, True]
    assert (config.mla_qk_nope_head_dim, config.qk_nope_head_dim) == (192, 448)
    assert config.mla_v_head_dim == 256
    assert config.rope_parameters["rope_type"] == "default"
    assert config.head_dtype == "float32"


def test_architectures_and_config_registered():
    supported = ModelRegistry.get_supported_archs()
    assert set(ernie5_5.ARCHITECTURES) <= set(supported)
    assert _CONFIG_REGISTRY["ernie5_5_moe"] is Ernie5_5Config
    breakable = default_breakable_cudagraph_architectures()
    assert {"Ernie5_5_MoeForCausalLM", "Ernie5_5MoeMTPModel"} <= breakable


def test_is_deepseek_mla():
    config = Ernie5_5Config(num_hidden_layers=2, compress_ratios=[128, -2])
    for model_type in ("ernie5_5_moe", "ernie5_5_moe_mtp"):
        convertor = MODEL_ARCH_CONFIG_CONVERTORS[model_type](config, config)
        assert convertor.is_deepseek_mla()


def test_mtp_hf_config_override():
    assert "ernie5_5_moe_mtp" in get_args(spec_mod.MTPModelTypes)
    config = Ernie5_5Config(num_hidden_layers=2, num_nextn_predict_layers=1)
    config.architectures = ["Ernie5_5_MoeForCausalLM"]
    config = SpeculativeConfig.hf_config_override(config)
    assert config.model_type == "ernie5_5_moe_mtp"
    assert config.architectures == ["Ernie5_5MoeMTPModel"]
    assert config.n_predict == 1

    # The draft ModelConfig carries this function to engine-core processes.
    fn = pickle.loads(pickle.dumps(SpeculativeConfig.hf_config_override))
    assert fn is SpeculativeConfig.hf_config_override


def test_dspark_draft_keeps_own_config():
    config = Ernie5_5Config(num_hidden_layers=2)
    config.architectures = ["Ernie5_5_MoeDSparkDraftModel"]
    assert SpeculativeConfig.hf_config_override(config).model_type == "ernie5_5_moe"

    # State after the DeepSeek V4 DSpark rewrite in SpeculativeConfig.
    verified = []
    draft = SimpleNamespace(
        architectures=["Ernie5_5_MoeDSparkDraftModel"],
        hf_config=SimpleNamespace(
            model_type="deepseek_v4", architectures=["DSparkDraftModel"]
        ),
        quantization="fp8",
        _verify_quantization=lambda: verified.append(True),
    )
    spec = SimpleNamespace(method="dspark", draft_model_config=draft, quantization=None)
    _undo_dspark_rewrite(spec)
    assert draft.hf_config.model_type == "ernie5_5_moe"
    assert draft.hf_config.architectures == ["Ernie5_5_MoeDSparkDraftModel"]
    assert draft.quantization is None
    assert verified
