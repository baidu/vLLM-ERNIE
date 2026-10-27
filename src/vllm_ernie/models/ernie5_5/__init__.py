# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ERNIE 5.5 (``Ernie5_5_MoeForCausalLM``) registration.

Kept light: the model code in ``nvidia/`` is only imported when the model
registry resolves one of these architectures.
"""

_MODEL_MODULE = "vllm_ernie.models.ernie5_5.nvidia"

ARCHITECTURES = {
    "Ernie5_5_MoeForCausalLM": "Ernie5_5ForCausalLM",
    "Ernie5_5MoeMTPModel": "Ernie5_5MTP",
    "Ernie5_5_MoeDSparkDraftModel": "Ernie5_5DSparkForCausalLM",
}


def register() -> None:
    from vllm import ModelRegistry

    for arch, cls_name in ARCHITECTURES.items():
        if arch not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(arch, f"{_MODEL_MODULE}:{cls_name}")

    from .patches import apply_patches

    apply_patches()
