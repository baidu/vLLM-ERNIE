# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hooks that patch shared vLLM code for ERNIE 5.5."""

import functools
import inspect
from typing import Literal, get_args

from vllm.logger import init_logger

logger = init_logger(__name__)

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
    # Restore the ERNIE DSpark draft config rewritten by __post_init__.
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


_MOONCAKE = "MooncakeConnector"


def _patch_mooncake_scheduler(scheduler_cls) -> None:
    """Send the KV of a prefill that stopped on its one token (vllm#63998)."""
    from vllm.v1.request import RequestStatus

    orig = scheduler_cls.request_finished
    if getattr(orig, "_ernie5_stopped_prefill", False):
        return
    # A no-op once upstream accepts FINISHED_STOPPED; only hint at removal.
    try:
        fixed = "!= RequestStatus.FINISHED_LENGTH_CAPPED" not in inspect.getsource(orig)
    except (OSError, TypeError):
        fixed = False
    if fixed:
        logger.info_once(
            "MooncakeConnector seems to send stopped prefills already; the "
            "ERNIE workaround in vllm_ernie patches.py can be removed."
        )

    @functools.wraps(orig)
    def request_finished(self, request, block_ids):
        params = request.kv_transfer_params
        if (
            request.status != RequestStatus.FINISHED_STOPPED
            or not params
            or not params.get("do_remote_decode")
            or self.vllm_config.model_config.hf_config.model_type != ERNIE5_MODEL_TYPE
        ):
            return orig(self, request, block_ids)
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        try:
            return orig(self, request, block_ids)
        finally:
            request.status = RequestStatus.FINISHED_STOPPED

    request_finished._ernie5_stopped_prefill = True  # type: ignore[attr-defined]
    scheduler_cls.request_finished = request_finished


def _patch_mooncake_connector() -> None:
    # Patch lazily, only when a server loads the Mooncake connector.
    from vllm.distributed.kv_transfer.kv_connector.factory import (
        KVConnectorFactory,
    )

    loader = KVConnectorFactory._registry.get(_MOONCAKE)
    if loader is None or getattr(loader, "_ernie5_patched", False):
        return

    def patched_loader():
        connector_cls = loader()
        module = inspect.getmodule(connector_cls)
        scheduler_cls = getattr(module, "MooncakeConnectorScheduler", None)
        if scheduler_cls is None:
            logger.warning_once(
                "MooncakeConnectorScheduler not found; the ERNIE stopped "
                "prefill workaround is not applied."
            )
        else:
            _patch_mooncake_scheduler(scheduler_cls)
        return connector_cls

    patched_loader._ernie5_patched = True  # type: ignore[attr-defined]
    KVConnectorFactory._registry[_MOONCAKE] = patched_loader


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
    _patch_mooncake_connector()
