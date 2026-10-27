# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM general plugin for ERNIE 5.5 (``Ernie5_5_MoeForCausalLM``).

``register()`` runs in every vLLM process, so this module stays light: model
code is only imported when the model registry resolves an architecture.
"""

_MODEL_MODULE = "vllm_ernie5_plugin.ernie5"

ARCHITECTURES = {
    "Ernie5_5_MoeForCausalLM": "Ernie5_5ForCausalLM",
    "Ernie5_5MoeMTPModel": "Ernie5_5MTP",
    "Ernie5_5_MoeDSparkDraftModel": "Ernie5_5DSparkForCausalLM",
}

_registered = False


def register() -> None:
    global _registered
    if _registered:
        return
    _registered = True

    from vllm import ModelRegistry

    for arch, cls_name in ARCHITECTURES.items():
        if arch not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(arch, f"{_MODEL_MODULE}:{cls_name}")

    from vllm_ernie5_plugin.patches import apply_patches

    apply_patches()
