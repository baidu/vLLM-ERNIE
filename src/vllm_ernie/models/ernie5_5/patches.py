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
ERNIE5_TOKENIZER_MODE = "ernie5_5"
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
    from vllm_ernie.models.ernie5_5.config import Ernie5_5Config

    draft = spec_config.draft_model_config
    hf_config = draft.hf_config
    if spec_config.method != "dspark" or hf_config.architectures != [
        "DSparkDraftModel"
    ]:
        return
    if ERNIE5_DSPARK_ARCH not in draft.architectures:
        # Without a draft path the draft config is the target's.
        if not (
            isinstance(hf_config, Ernie5_5Config) and hf_config.bundles_dspark_draft
        ):
            return
        hf_config.to_bundled_dspark_draft()
    hf_config.model_type = ERNIE5_MODEL_TYPE
    hf_config.architectures = [ERNIE5_DSPARK_ARCH]
    # The rewrite also copied the target quantization; re-derive the
    # draft's own from its checkpoint.
    draft.quantization = spec_config.quantization
    draft._verify_quantization()


def _default_bundled_dspark_tokens(spec_config) -> None:
    # vLLM only reads the block size from a draft config; without a draft path
    # it needs num_speculative_tokens first, so take it from the target.
    target = spec_config.target_model_config
    if (
        spec_config.method == "dspark"
        and spec_config.model is None
        and spec_config.num_speculative_tokens is None
        and target is not None
        and getattr(target.hf_config, "bundles_dspark_draft", False)
    ):
        spec_config.num_speculative_tokens = target.hf_config.dspark_block_size


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

    orig_post_init = SpeculativeConfig.__post_init__

    @functools.wraps(orig_post_init)
    def __post_init__(self):
        _default_bundled_dspark_tokens(self)
        orig_post_init(self)

    SpeculativeConfig.__post_init__ = __post_init__


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


def _register_renderer() -> None:
    from vllm.renderers.registry import RENDERER_REGISTRY, RendererRegistry
    from vllm.tokenizers.registry import TokenizerRegistry

    TokenizerRegistry.register(
        ERNIE5_TOKENIZER_MODE, "vllm.tokenizers.hf", "CachedHfTokenizer"
    )
    RENDERER_REGISTRY.register(
        ERNIE5_TOKENIZER_MODE,
        "vllm_ernie.models.ernie5_5.renderer",
        "Ernie5_5Renderer",
    )

    # Use the ERNIE renderer by default; it falls back to HfRenderer when the
    # checkpoint ships a jinja template.
    orig = RendererRegistry.load_renderer
    if getattr(orig, "_ernie5_default_renderer", False):
        return

    @functools.wraps(orig)
    def load_renderer(self, renderer_mode, config, tokenizer):
        if (
            renderer_mode == "hf"
            and config.model_config.hf_config.model_type == ERNIE5_MODEL_TYPE
        ):
            renderer_mode = ERNIE5_TOKENIZER_MODE
        return orig(self, renderer_mode, config, tokenizer)

    load_renderer._ernie5_default_renderer = True  # type: ignore[attr-defined]
    RendererRegistry.load_renderer = load_renderer


def _with_parallel_tool_calls_option(request) -> None:
    """Pass ``parallel_tool_calls=false`` to the chat template as an option.

    The ERNIE template reads ``options.parallel_tool_calls`` and announces
    parallel calls unless it is false; ``ChatParams`` has no such field.
    """
    if not getattr(request, "tools", None):
        return
    if getattr(request, "parallel_tool_calls", True) is not False:
        return
    kwargs = dict(request.chat_template_kwargs or {})
    options = dict(kwargs.get("options") or {})
    options.setdefault("parallel_tool_calls", False)
    kwargs["options"] = options
    request.chat_template_kwargs = kwargs


def _patch_parallel_tool_calls() -> None:
    from vllm.renderers.online_renderer import OnlineRenderer

    orig = OnlineRenderer.preprocess_chat
    if getattr(orig, "_ernie5_parallel_tool_calls", False):
        return

    @functools.wraps(orig)
    async def preprocess_chat(self, request, *args, **kwargs):
        if self.model_config.hf_config.model_type == ERNIE5_MODEL_TYPE:
            _with_parallel_tool_calls_option(request)
        return await orig(self, request, *args, **kwargs)

    preprocess_chat._ernie5_parallel_tool_calls = True  # type: ignore[attr-defined]
    OnlineRenderer.preprocess_chat = preprocess_chat


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
    _register_renderer()
    _patch_parallel_tool_calls()
    _patch_mooncake_connector()
