# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hooks that reproduce the ERNIE 5.5 changes to shared vLLM code.

Each hook mirrors one in-tree change on ``eblite-v1``:

- ``_CONFIG_REGISTRY["ernie5_5_moe"] = Ernie5_5Config``
- ``Ernie5_5_MoeForCausalLM`` / ``Ernie5_5MoeMTPModel`` use breakable CUDA
  graphs
- ``is_deepseek_mla`` covers ``ernie5_5_moe`` and ``ernie5_5_moe_mtp``;
  ``ernie5_5_moe_mtp`` uses the DeepSeek MTP arch convertor
- speculative config: ``ernie5_5_moe_mtp`` is an MTP model type, the target
  config is rewritten to ``Ernie5_5MoeMTPModel`` for MTP, and the ERNIE DSpark
  draft keeps its own config instead of the DeepSeek V4 DSpark rewrite
- ``ernie5`` reasoning and tool-call parsers
"""

from typing import Literal, get_args

ERNIE5_MODEL_TYPE = "ernie5_5_moe"
ERNIE5_MTP_MODEL_TYPE = "ernie5_5_moe_mtp"
ERNIE5_MTP_ARCH = "Ernie5_5MoeMTPModel"
ERNIE5_DSPARK_ARCH = "Ernie5_5_MoeDSparkDraftModel"
ERNIE5_BREAKABLE_CUDAGRAPH_ARCHITECTURES = (
    "Ernie5_5MoeMTPModel",
    "Ernie5_5_MoeForCausalLM",
)

_applied = False


def _patch_config_registry() -> None:
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    from vllm_ernie.models.ernie5_5.config import Ernie5_5Config

    _CONFIG_REGISTRY.setdefault(ERNIE5_MODEL_TYPE, Ernie5_5Config)


def _patch_breakable_cudagraph() -> None:
    import vllm.config.vllm as vllm_config_mod

    vllm_config_mod.DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES = frozenset(
        vllm_config_mod.DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES
        | set(ERNIE5_BREAKABLE_CUDAGRAPH_ARCHITECTURES)
    )
    vllm_config_mod.default_breakable_cudagraph_architectures.cache_clear()


def _is_ernie5_deepseek_mla(hf_text_config) -> bool:
    if hasattr(hf_text_config, "compress_ratios"):
        return getattr(hf_text_config, "head_dim", None) is not None
    return getattr(hf_text_config, "kv_lora_rank", None) is not None


def _patch_arch_config_convertors() -> None:
    from vllm.transformers_utils.model_arch_config_convertor import (
        MODEL_ARCH_CONFIG_CONVERTORS,
        DeepSeekMTPModelArchConfigConvertor,
        ModelArchConfigConvertorBase,
    )

    class Ernie5_5ModelArchConfigConvertor(ModelArchConfigConvertorBase):
        def is_deepseek_mla(self) -> bool:
            return _is_ernie5_deepseek_mla(self.hf_text_config)

    class Ernie5_5MoeMTPModelArchConfigConvertor(DeepSeekMTPModelArchConfigConvertor):
        def is_deepseek_mla(self) -> bool:
            return _is_ernie5_deepseek_mla(self.hf_text_config)

    MODEL_ARCH_CONFIG_CONVERTORS.setdefault(
        ERNIE5_MODEL_TYPE, Ernie5_5ModelArchConfigConvertor
    )
    MODEL_ARCH_CONFIG_CONVERTORS.setdefault(
        ERNIE5_MTP_MODEL_TYPE, Ernie5_5MoeMTPModelArchConfigConvertor
    )


def _ernie5_hf_config_override(hf_config):
    if (
        hf_config.model_type == ERNIE5_MODEL_TYPE
        and hf_config.architectures[0] != ERNIE5_DSPARK_ARCH
    ):
        hf_config.model_type = ERNIE5_MTP_MODEL_TYPE
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update({"n_predict": n_predict, "architectures": [ERNIE5_MTP_ARCH]})
    return hf_config


def _undo_dspark_rewrite(spec_config) -> None:
    # The DeepSeek V4 DSpark branch of __post_init__ rewrites every DSpark
    # draft it does not know; undo it for the self-contained ERNIE draft.
    # ``draft.architectures`` still holds the pre-rewrite value here since
    # model_arch_config is only rebuilt by update_arch_.
    draft = spec_config.draft_model_config
    if (
        spec_config.method == "dspark"
        and ERNIE5_DSPARK_ARCH in draft.architectures
        and draft.hf_config.architectures == ["DSparkDraftModel"]
    ):
        draft.hf_config.model_type = ERNIE5_MODEL_TYPE
        draft.hf_config.architectures = [ERNIE5_DSPARK_ARCH]
        # The rewrite also copied the target quantization; re-derive the
        # draft's own from its checkpoint.
        draft.quantization = spec_config.quantization
        draft._verify_quantization()


def _patch_speculative_config() -> None:
    import vllm.config.speculative as spec_mod
    from vllm.config.speculative import SpeculativeConfig

    if ERNIE5_MTP_MODEL_TYPE not in get_args(spec_mod.MTPModelTypes):
        spec_mod.MTPModelTypes = Literal[  # type: ignore[misc]
            (*get_args(spec_mod.MTPModelTypes), ERNIE5_MTP_MODEL_TYPE)
        ]

    orig_override = SpeculativeConfig.hf_config_override

    def hf_config_override(hf_config):
        return orig_override(_ernie5_hf_config_override(hf_config))

    # compose_draft_hf_overrides hands this function to the draft ModelConfig,
    # which is pickled to engine-core processes; keep it importable by name.
    hf_config_override.__module__ = SpeculativeConfig.__module__
    hf_config_override.__qualname__ = "SpeculativeConfig.hf_config_override"
    hf_config_override.__wrapped__ = orig_override  # type: ignore[attr-defined]
    SpeculativeConfig.hf_config_override = staticmethod(hf_config_override)

    orig_update_arch = SpeculativeConfig.update_arch_

    def update_arch_(self):
        _undo_dspark_rewrite(self)
        orig_update_arch(self)

    SpeculativeConfig.update_arch_ = update_arch_


def _register_parsers() -> None:
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    ReasoningParserManager.register_lazy_module(
        "ernie5",
        "vllm_ernie.models.ernie5_5.reasoning_parser",
        "Ernie5ParserReasoningAdapter",
    )
    ToolParserManager.register_lazy_module(
        "ernie5", "vllm_ernie.models.ernie5_5.tool_parser", "Ernie5ToolParser"
    )


def apply_patches() -> None:
    global _applied
    if _applied:
        return
    _applied = True
    _patch_config_registry()
    _patch_breakable_cudagraph()
    _patch_arch_config_convertors()
    _patch_speculative_config()
    _register_parsers()
