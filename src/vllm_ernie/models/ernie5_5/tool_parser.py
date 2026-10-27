# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xgrammar import StructuralTag
from xgrammar.structural_tag import (
    ConstStringFormat,
    JSONSchemaFormat,
    SequenceFormat,
    TagFormat,
    TagsWithSeparatorFormat,
    TriggeredTagsFormat,
)

from vllm.tool_parsers.structural_tag_registry import register_vllm_structural_tag
from vllm.tool_parsers.tool_strict_level import ToolStrictLevel

from vllm_ernie.models.ernie5_5.parser import (
    TOOL_CALL_END,
    TOOL_CALL_START,
    Ernie5ParserToolAdapter,
)

if TYPE_CHECKING:
    from xgrammar.openai_tool_call_schema import BuiltinToolParam, FunctionToolParam

    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tool_parsers.structural_tag_registry import SimplifiedToolChoice

_EMPTY_PARAMETERS = {"type": "object", "properties": {}}


@register_vllm_structural_tag("ernie5")
def get_ernie5_structural_tag(
    tools: list[FunctionToolParam],
    builtin_tools: list[BuiltinToolParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
    token_suffix: str = "",
) -> StructuralTag:
    """Build the ``<ernie:tool_call>`` structural tag.

    Uses ``<invoke>`` elements with no newline after ``">``, and the argument
    schema is always pinned regardless of ``strict``.

    Args:
        tools: Normalized function tools the model may call.
        builtin_tools: Unused; ERNIE 5 has no builtin tools.
        tool_choice: Simplified tool choice.
        reasoning: Unused; the reasoning parser owns the think section.
        token_suffix: Unused.

    """
    del builtin_tools, reasoning, token_suffix

    invokes = TagsWithSeparatorFormat(
        tags=[
            TagFormat(
                begin=f'<invoke name="{tool.function.name}">',
                content=JSONSchemaFormat(
                    json_schema=tool.function.parameters or _EMPTY_PARAMETERS,
                    style="minimax_xml",
                ),
                end="</invoke>\n",
            )
            for tool in tools
        ],
        separator="",
        at_least_one=True,
    )
    if tool_choice == "auto":
        body: Any = TriggeredTagsFormat(
            triggers=[TOOL_CALL_START],
            tags=[
                TagFormat(
                    begin=TOOL_CALL_START + "\n", content=invokes, end=TOOL_CALL_END
                )
            ],
        )
    else:
        body = SequenceFormat(
            elements=[
                ConstStringFormat(value=TOOL_CALL_START + "\n"),
                invokes,
                ConstStringFormat(value=TOOL_CALL_END),
            ]
        )
    return StructuralTag(format=body)


class Ernie5ToolParser(Ernie5ParserToolAdapter):  # type: ignore[valid-type, misc]
    supports_required_and_named = False
    structural_tag_model = "ernie5"

    def get_structural_tag(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
        *,
        reasoning: bool = False,
        strict_level: ToolStrictLevel = ToolStrictLevel.AUTO,
    ) -> StructuralTag | None:
        tag = super().get_structural_tag(
            request, reasoning=reasoning, strict_level=strict_level
        )
        if tag is None or getattr(request, "parallel_tool_calls", True) is not False:
            return tag
        # parallel_tool_calls=false allows a single invoke.
        fmt = tag.format
        invokes = (
            fmt.tags[0].content
            if isinstance(fmt, TriggeredTagsFormat)
            else fmt.elements[1]  # type: ignore[union-attr]
        )
        invokes.stop_after_first = True
        return tag
