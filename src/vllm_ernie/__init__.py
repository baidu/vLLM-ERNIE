# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM general plugin for ERNIE models.

``register()`` runs in every vLLM process, so it only registers each model
under ``models/``; model code is imported when an architecture resolves.
"""

import logging

_registered = False


def register() -> None:
    global _registered
    if _registered:
        return
    _registered = True

    # vLLM configures only the `vllm` logger; share its handlers so the
    # plugin's INFO logs are not dropped.
    vllm_logger = logging.getLogger("vllm")
    plugin_logger = logging.getLogger(__name__)
    if not plugin_logger.handlers:
        plugin_logger.handlers = list(vllm_logger.handlers)
        plugin_logger.setLevel(vllm_logger.level)
        plugin_logger.propagate = False

    from vllm_ernie.models import ernie5_5

    ernie5_5.register()
