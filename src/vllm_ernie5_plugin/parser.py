# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ERNIE 5 parser for reasoning and tool calls.

The chat template ends the generation prompt with ``<ernie:think>`` (thinking
on, the default) or ``<ernie:think></ernie:think>`` (thinking off), so the
model output is::

    reasoning</ernie:think>content
    <ernie:tool_call>
    <invoke name="get_weather"><parameter name="city">Paris</parameter></invoke>
    <invoke name="get_time"><parameter name="tz">CET</parameter></invoke>
    </ernie:tool_call>

The tool-call body is a list of ``<invoke>`` elements. For parameter
values, ``null``/``none``/``nil`` become null, ``yes``/``on``/``no``/``off`` are
booleans, local ``$ref`` is resolved, and the rest is coerced by the tool schema.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from vllm.entrypoints.generate.base.protocol import ExtractedToolCallInformation
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
from vllm.parser.engine.adapters import make_adapters
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import ParserEngineConfig, ParserState
from vllm.parser.minimax_m2 import (
    _PARAM_RE,
    _PARTIAL_PARAM_RE,
    _minimax_m2_arg_converter,
    minimax_m2_config,
)
from vllm.tool_parsers.utils import (
    coerce_to_schema_type,
    extract_types_from_schema,
    iter_response_function_tool_info,
)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

THINK_START = "<ernie:think>"
THINK_END = "</ernie:think>"
TOOL_CALL_START = "<ernie:tool_call>"
TOOL_CALL_END = "</ernie:tool_call>"
TURN_BOUNDARIES = frozenset(("<|ernie:im_start|>", "<|ernie:im_end|>"))
_THINKING_OFF = (False, "false", "False")
_THINKING_MODE_OFF = ("close", "false")
_NULL_WORDS = ("null", "none", "nil")
_BOOL_WORDS = {"yes": True, "on": True, "no": False, "off": False}
_BOOL_TYPES = frozenset(("boolean", "bool"))
_COMBINATORS = ("anyOf", "oneOf", "allOf")


def _lookup_pointer(root: Any, ref: str) -> Any:
    node = root
    for part in ref[1:].split("/")[1:]:
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict):
            node = node.get(part)
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return None
    return node


def _resolve_refs(schema: Any, root: Any, seen: tuple[str, ...] = ()) -> Any:
    """Inline local ``$ref`` on the node and its anyOf/oneOf/allOf branches."""
    if not isinstance(schema, dict):
        return schema
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#") and ref not in seen:
        target = _lookup_pointer(root, ref)
        if isinstance(target, dict):
            siblings = {k: v for k, v in schema.items() if k != "$ref"}
            return _resolve_refs({**target, **siblings}, root, (*seen, ref))
    resolved = dict(schema)
    for keyword in _COMBINATORS:
        if isinstance(schema.get(keyword), list):
            resolved[keyword] = [_resolve_refs(b, root, seen) for b in schema[keyword]]
    return resolved


def _find_tool_parameters(tools: list[Tool] | None, name: str) -> dict[str, Any]:
    """Full parameters schema of tool ``name``, so ``$defs`` stay reachable."""
    for tool in tools or ():
        if isinstance(tool, ChatCompletionToolsParam):
            infos = [(tool.function.name, tool.function.parameters)]
        else:
            infos = iter_response_function_tool_info(tool)
        for tool_name, params in infos:
            if tool_name == name:
                return params or {}
    return {}


def _has_ref(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    return "$ref" in schema or any(
        _has_ref(branch)
        for keyword in _COMBINATORS
        if isinstance(schema.get(keyword), list)
        for branch in schema[keyword]
    )


def _schema_properties(schema: Any) -> dict[str, Any]:
    """Top-level properties, merged from anyOf/oneOf/allOf when absent."""
    if not isinstance(schema, dict):
        return {}
    properties = schema.get("properties")
    if isinstance(properties, dict):
        return properties
    merged: dict[str, Any] = {}
    for keyword in _COMBINATORS:
        if isinstance(schema.get(keyword), list):
            for branch in schema[keyword]:
                for key, value in _schema_properties(branch).items():
                    merged.setdefault(key, value)
    return merged


def coerce_param(value: str, schema: Any) -> Any:
    """Convert a raw ``<parameter>`` value using the tool schema."""
    if value.lower() in _NULL_WORDS:
        return None
    types = extract_types_from_schema(schema)
    coerced = coerce_to_schema_type(value, types)
    if coerced is value and _BOOL_TYPES & {t.lower() for t in types}:
        return _BOOL_WORDS.get(value.strip().lower(), value)
    return coerced


def _ernie5_arg_converter(raw_args: str, partial: bool) -> str:
    args_json = _minimax_m2_arg_converter(raw_args, partial)
    if not partial:
        return args_json
    # Hold back an open value that may still become null, so a streamed
    # string prefix never has to turn into ``null``.
    match = _PARTIAL_PARAM_RE.search(_PARAM_RE.sub("", raw_args))
    if match is None:
        return args_json
    value = match.group("value").lower()
    if not any(word.startswith(value) for word in _NULL_WORDS):
        return args_json
    name = (
        match.group("dq_name") or match.group("sq_name") or match.group("bare_name")
    ).strip()
    params = json.loads(args_json)
    params.pop(name, None)
    return json.dumps(params, ensure_ascii=False)


@functools.cache
def ernie5_config(thinking: bool = True) -> ParserEngineConfig:
    base = minimax_m2_config()
    tokens = {
        "THINK_START": THINK_START,
        "THINK_END": THINK_END,
        "TOOL_START": TOOL_CALL_START,
        "TOOL_END": TOOL_CALL_END,
    }
    return replace(
        base,
        name="ernie5",
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        wait_for_reasoning=thinking,
        terminals={**base.terminals, **tokens},
        token_id_terminals={**base.token_id_terminals, **tokens},
        turn_boundary_tokens=TURN_BOUNDARIES,
        arg_converter=_ernie5_arg_converter,
    )


def is_thinking_enabled(chat_template_kwargs: dict | None) -> bool:
    """Mirror the ERNIE 5 chat template's thinking switch."""
    kwargs = chat_template_kwargs or {}
    options = kwargs.get("options") or {}
    return (
        kwargs.get("enable_thinking", True) not in _THINKING_OFF
        and options.get("thinking_mode") not in _THINKING_MODE_OFF
    )


class Ernie5Parser(ParserEngine):
    """ERNIE 5 parser backed by the XML parser engine."""

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        self.thinking_enabled = is_thinking_enabled(kwargs.get("chat_template_kwargs"))
        kwargs.setdefault("parser_engine_config", ernie5_config(self.thinking_enabled))
        super().__init__(tokenizer, tools, **kwargs)

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if not self.thinking_enabled:
            return None, model_output
        return super().extract_reasoning(model_output, request)

    def _single_pass_parse(
        self,
        text: str,
        token_ids: Sequence[int],
        initial_state: ParserState | None = None,
    ) -> tuple[str | None, str | None, ExtractedToolCallInformation]:
        reasoning, content, tool_call_info = super()._single_pass_parse(
            text, token_ids, initial_state=initial_state
        )
        # Keep text after </ernie:tool_call>, matching streaming.
        if not self._deferred_content:
            return reasoning, content, tool_call_info
        trailing = self._deferred_content
        self._deferred_content = ""
        content = self._strip_content_whitespace(
            (content or "") + trailing, tool_call_info.tools_called
        )
        return (
            reasoning,
            content,
            ExtractedToolCallInformation(
                tools_called=tool_call_info.tools_called,
                tool_calls=tool_call_info.tool_calls,
                content=content,
            ),
        )

    def _fix_arg_types(self, args_json: str, func_name: str) -> str:
        if not func_name:
            return args_json
        try:
            args = json.loads(args_json)
        except ValueError:
            return args_json
        if not isinstance(args, dict):
            return args_json
        root = _find_tool_parameters(self._tools, func_name)
        properties = _schema_properties(root)
        for key, value in args.items():
            if isinstance(value, str):
                schema = _resolve_refs(properties.get(key), root)
                args[key] = coerce_param(value, schema)
        return json.dumps(args, ensure_ascii=False)

    @staticmethod
    def _streamable_string_keys(properties: dict) -> set[str] | None:
        keys = ParserEngine._streamable_string_keys(properties)
        if keys is None:
            return None
        return {key for key in keys if not _has_ref(properties[key])}


(
    Ernie5ParserReasoningAdapter,
    Ernie5ParserToolAdapter,
) = make_adapters(Ernie5Parser)
