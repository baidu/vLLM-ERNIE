# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the engine-based ERNIE 5 parser.

These tests cover what is ERNIE 5 specific: the ``<ernie:tool_call>`` wrappers,
the chat-template thinking switches, the ``<|ernie:im_*|>`` turn scoping of
``is_reasoning_end``, and the registered adapters.
"""

import json
from unittest.mock import MagicMock

import pytest
from parser_helpers import (
    collect_function_name,
    collect_tool_arguments,
    make_mock_tokenizer,
    simulate_reasoning_streaming,
    simulate_tool_streaming,
)
from xgrammar import Grammar, StructuralTag
from xgrammar.testing import _is_grammar_accept_string

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag

from vllm_ernie.models.ernie5_5.parser import (
    THINK_END,
    THINK_START,
    TOOL_CALL_END,
    TOOL_CALL_START,
    Ernie5Parser,
    Ernie5ParserReasoningAdapter,
)
from vllm_ernie.models.ernie5_5.tool_parser import Ernie5ToolParser

_IM_START_ID = 7
_IM_END_ID = 8
_TOOL_CALL_START_ID = 17
_TOOL_CALL_END_ID = 18
_THINK_START_ID = 19
_THINK_END_ID = 20
_TEXT_ID = 100

_TOOL_OUTPUT = (
    f"{TOOL_CALL_START}\n"
    '<invoke name="get_weather"><parameter name="city">Paris</parameter>'
    '<parameter name="days">3</parameter></invoke>\n'
    '<invoke name="get_time"><parameter name="tz">CET</parameter></invoke>\n'
    f"{TOOL_CALL_END}"
)


@pytest.fixture
def mock_tokenizer():
    return make_mock_tokenizer(
        {
            "<|ernie:im_start|>": _IM_START_ID,
            "<|ernie:im_end|>": _IM_END_ID,
            TOOL_CALL_START: _TOOL_CALL_START_ID,
            TOOL_CALL_END: _TOOL_CALL_END_ID,
            THINK_START: _THINK_START_ID,
            THINK_END: _THINK_END_ID,
        }
    )


@pytest.fixture
def parser(mock_tokenizer):
    return Ernie5Parser(mock_tokenizer)


@pytest.fixture
def tools():
    return [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"},
                    },
                },
            ),
        ),
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_time",
                parameters={
                    "type": "object",
                    "properties": {"tz": {"type": "string"}},
                },
            ),
        ),
    ]


def test_adapters_registered():
    assert ReasoningParserManager.get_reasoning_parser("ernie5")
    assert ToolParserManager.get_tool_parser("ernie5")


def test_reasoning_and_content(parser, mock_request):
    reasoning, content = parser.extract_reasoning(
        f"Let me think.{THINK_END}The answer is 4.", mock_request
    )
    assert reasoning == "Let me think."
    assert content == "The answer is 4."


@pytest.mark.parametrize(
    "chat_template_kwargs",
    [
        {"enable_thinking": False},
        {"enable_thinking": "false"},
        {"options": {"thinking_mode": "close"}},
    ],
)
def test_thinking_off(mock_tokenizer, mock_request, chat_template_kwargs):
    parser = Ernie5Parser(mock_tokenizer, chat_template_kwargs=chat_template_kwargs)
    assert not parser.thinking_enabled
    assert parser.extract_reasoning("Plain answer.", mock_request) == (
        None,
        "Plain answer.",
    )
    assert parser.is_reasoning_end([_IM_START_ID, _TEXT_ID])


def test_reasoning_then_parallel_tool_calls(mock_tokenizer, mock_request, tools):
    reasoning_parser = Ernie5ParserReasoningAdapter(mock_tokenizer)
    tool_parser = Ernie5ToolParser(mock_tokenizer, tools)
    mock_request.tools = tools
    text = f"Need weather and time.{THINK_END}Checking.\n{_TOOL_OUTPUT}"

    reasoning, content = reasoning_parser.extract_reasoning(text, mock_request)
    assert reasoning == "Need weather and time."
    result = tool_parser.extract_tool_calls(content, mock_request)

    assert result.tools_called
    assert result.content == "Checking."
    assert [tc.function.name for tc in result.tool_calls] == [
        "get_weather",
        "get_time",
    ]
    assert json.loads(result.tool_calls[0].function.arguments) == {
        "city": "Paris",
        "days": 3,
    }
    assert json.loads(result.tool_calls[1].function.arguments) == {"tz": "CET"}


def test_tool_call_ends_reasoning(mock_tokenizer, mock_request):
    reasoning_parser = Ernie5ParserReasoningAdapter(mock_tokenizer)
    reasoning, content = reasoning_parser.extract_reasoning(
        f"Calling a tool.{_TOOL_OUTPUT}", mock_request
    )
    assert reasoning == "Calling a tool."
    assert content.startswith(TOOL_CALL_START)


def test_parse_keeps_text_after_tool_call(mock_tokenizer, mock_request, tools):
    """Non-streaming ``parse`` keeps trailing text, matching streaming."""
    parser = Ernie5Parser(mock_tokenizer, tools=tools)
    mock_request.tools = tools
    reasoning, content, tool_calls = parser.parse(
        f"Think.{THINK_END}\n{_TOOL_OUTPUT}\nAfter.",
        mock_request,
        enable_auto_tools=True,
    )
    assert reasoning == "Think."
    assert content == "After."
    assert [tc.name for tc in tool_calls] == ["get_weather", "get_time"]


_CONVERT_TOOL = ChatCompletionToolsParam(
    function=FunctionDefinition(
        name="f",
        parameters={
            "type": "object",
            "$defs": {"N": {"type": "integer"}, "Slot": {"type": "object"}},
            "properties": {
                "s": {"type": "string"},
                "b": {"type": "boolean"},
                "n": {"$ref": "#/$defs/N"},
                "slot": {"$ref": "#/$defs/Slot"},
                "u": {"anyOf": [{"$ref": "#/$defs/N"}, {"type": "boolean"}]},
            },
        },
    ),
)


def _invoke(**params) -> str:
    body = "".join(f'<parameter name="{k}">{v}</parameter>' for k, v in params.items())
    return f'{TOOL_CALL_START}\n<invoke name="f">{body}</invoke>\n{TOOL_CALL_END}'


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        # null/none/nil are null for every type, string included.
        ({"s": "null", "b": "None", "n": "nil"}, {"s": None, "b": None, "n": None}),
        (
            {"b": "yes"},
            {"b": True},
        ),
        ({"b": "Off"}, {"b": False}),
        ({"b": "maybe"}, {"b": "maybe"}),
        # Local $ref, also inside anyOf, is resolved before type coercion.
        ({"n": "5", "slot": '{"start": "9:00"}'}, {"n": 5, "slot": {"start": "9:00"}}),
        ({"u": "on"}, {"u": True}),
        ({"s": "yes"}, {"s": "yes"}),
    ],
)
def test_param_conversion(mock_tokenizer, mock_request, params, expected):
    """Values convert the same way in both parse paths."""
    mock_request.tools = [_CONVERT_TOOL]
    text = _invoke(**params)

    parser = Ernie5Parser(mock_tokenizer, tools=[_CONVERT_TOOL])
    result = parser.extract_tool_calls(text, mock_request)
    assert json.loads(result.tool_calls[0].function.arguments) == expected

    parser = Ernie5Parser(
        mock_tokenizer,
        tools=[_CONVERT_TOOL],
        chat_template_kwargs={"enable_thinking": False},
    )
    results = simulate_tool_streaming(parser, mock_request, list(text))
    assert json.loads(collect_tool_arguments(results)) == expected


class TestStreaming:
    def test_reasoning(self, parser):
        reasoning, content = simulate_reasoning_streaming(
            parser, ["Let me ", "think.", THINK_END, "Done."]
        )
        assert reasoning == "Let me think."
        assert content == "Done."

    def test_tool_calls(self, mock_tokenizer, mock_request, tools):
        parser = Ernie5Parser(
            mock_tokenizer,
            tools=tools,
            chat_template_kwargs={"enable_thinking": False},
        )
        mock_request.tools = tools
        results = simulate_tool_streaming(
            parser,
            mock_request,
            [
                TOOL_CALL_START,
                "\n<invoke name=",
                '"get_weather">',
                '<parameter name="city">Par',
                'is</parameter><parameter name="days">3</parameter>',
                "</invoke>\n",
                TOOL_CALL_END,
            ],
        )
        assert collect_function_name(results) == "get_weather"
        assert json.loads(collect_tool_arguments(results)) == {
            "city": "Paris",
            "days": 3,
        }


class TestTurnBoundaries:
    """``</ernie:think>`` from a replayed assistant turn must not end reasoning in
    the current turn."""

    def test_history_think_end_is_not_end(self, parser):
        assert not parser.is_reasoning_end(
            [
                _IM_START_ID,
                _THINK_START_ID,
                _TEXT_ID,
                _THINK_END_ID,
                _TEXT_ID,
                _IM_END_ID,
                _IM_START_ID,
                _TEXT_ID,
                _THINK_START_ID,
            ]
        )

    def test_think_end_in_current_turn_is_end(self, parser):
        assert parser.is_reasoning_end(
            [_IM_START_ID, _TEXT_ID, _THINK_END_ID, _TEXT_ID]
        )


def _call(*cities: str) -> str:
    invokes = "".join(
        f'<invoke name="get_weather"><parameter name="city">{c}</parameter></invoke>\n'
        for c in cities
    )
    return f"{TOOL_CALL_START}\n{invokes}{TOOL_CALL_END}"


def _grammar(tools, tool_choice="required", parallel=True) -> Grammar:
    request = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=[tool.model_dump() for tool in tools],
        tool_choice=tool_choice,
        parallel_tool_calls=parallel,
    )
    tag = Ernie5ToolParser(MagicMock(), tools).get_structural_tag(request)
    assert isinstance(tag, StructuralTag)
    return Grammar.from_structural_tag(tag)


class TestStructuralTag:
    """Constrained decoding emits the ``<ernie:tool_call>`` format."""

    def test_registered(self, tools):
        assert Ernie5ToolParser.structural_tag_model == "ernie5"
        assert not Ernie5ToolParser.supports_required_and_named
        tag = get_model_structural_tag("ernie5", tools, "required", reasoning=False)
        assert isinstance(tag, StructuralTag)

    @pytest.mark.parametrize(
        "tool_choice",
        ["required", {"type": "function", "function": {"name": "get_weather"}}],
    )
    def test_required_and_named_use_native_format(self, tools, tool_choice):
        # The parser only reads <ernie:tool_call>, so required/named must not
        # fall back to the JSON-array constraint.
        grammar = _grammar(tools, tool_choice)
        assert _is_grammar_accept_string(grammar, _call("Paris"))
        assert _is_grammar_accept_string(grammar, _call("Paris", "Rome"))
        assert not _is_grammar_accept_string(
            grammar, '[{"name": "get_weather", "parameters": {"city": "Paris"}}]'
        )
        # The argument schema is pinned even without ``strict``.
        assert not _is_grammar_accept_string(
            grammar,
            f'{TOOL_CALL_START}\n<invoke name="get_weather">'
            f'<parameter name="zip">1</parameter></invoke>\n{TOOL_CALL_END}',
        )

    def test_parallel_tool_calls_false_allows_one_invoke(self, tools):
        grammar = _grammar(tools, parallel=False)
        assert _is_grammar_accept_string(grammar, _call("Paris"))
        assert not _is_grammar_accept_string(grammar, _call("Paris", "Rome"))

    def test_auto_strict_allows_free_text(self, tools):
        strict = [
            t.model_copy(
                update={"function": t.function.model_copy(update={"strict": True})}
            )
            for t in tools
        ]
        grammar = _grammar(strict, "auto")
        assert _is_grammar_accept_string(grammar, "Just answering.")
        assert _is_grammar_accept_string(grammar, "Checking.\n" + _call("Paris"))
