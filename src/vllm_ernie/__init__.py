# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM general plugin for ERNIE models.

``register()`` runs in every vLLM process, so it only registers each model
under ``models/``; model code is imported when an architecture resolves.
"""

_registered = False


def register() -> None:
    global _registered
    if _registered:
        return
    _registered = True

    from vllm_ernie.models import ernie5_5

    ernie5_5.register()
