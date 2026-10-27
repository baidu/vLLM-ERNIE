# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ERNIE 5.5 hybrid MLA (``Ernie5_5_MoeForCausalLM``) model.

Built on the DeepSeek V4 stack (mHC residuals, C128 HCA attention) with a
few MLA + DSA layers, a latent MoE with split-feature routing and a multimax
output head. NVIDIA only.
"""

from vllm.platforms import current_platform

if current_platform.is_rocm() or current_platform.is_xpu():
    raise NotImplementedError("ERNIE 5.5 only supports NVIDIA GPUs.")

from .dspark import Ernie5_5DSparkForCausalLM  # noqa: E402
from .model import Ernie5_5ForCausalLM  # noqa: E402
from .mtp import Ernie5_5MTP  # noqa: E402

__all__ = [
    "Ernie5_5DSparkForCausalLM",
    "Ernie5_5ForCausalLM",
    "Ernie5_5MTP",
]
