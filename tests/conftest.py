# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

import vllm_ernie

# Engine processes get this through the ``vllm.general_plugins`` entry point;
# unit tests import plugin modules directly, so register up front.
vllm_ernie.register()


@pytest.fixture
def mock_request():
    req = MagicMock(spec=ChatCompletionRequest)
    req.tools = []
    req.tool_choice = "auto"
    req.include_reasoning = True
    return req
