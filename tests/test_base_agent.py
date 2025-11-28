"""
test_base_agent.py

Unit tests for the base agent loop.

These test the agentic loop mechanics without hitting the real Anthropic API
by using a mock client. We're testing:
- Tool dispatch works correctly
- The loop terminates on end_turn
- Tool errors are handled gracefully and returned to the model
- Max tool round limit is enforced
- Memory writes happen on successful completion
"""

import json
import pytest
from unittest.mock import MagicMock, patch, call
from dataclasses import dataclass
from typing import Any

from agents.base import BaseAgent, AgentResult


# ---------------------------------------------------------------------------
# Test double: a minimal concrete agent for testing
# ---------------------------------------------------------------------------

class _EchoAgent(BaseAgent):
    """Minimal agent that echoes tool inputs back. Used for testing the loop."""

    def __init__(self, memory_client=None):
        super().__init__(name="EchoAgent", memory_client=memory_client, model="claude-sonnet-4-6")
        self.tool_calls_received = []

    @property
    def system_prompt(self) -> str:
        return "You are a test agent."

    @property
    def tools(self) -> list[dict]:
        return [
            {
                "name": "echo",
                "description": "Echo back the input",
                "input_schema": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
            },
            {
                "name": "fail_tool",
                "description": "Always raises an exception",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]

    def execute_tool(self, tool_name: str, tool_input: dict) -> Any:
        self.tool_calls_received.append((tool_name, tool_input))
        if tool_name == "echo":
            return {"echoed": tool_input["message"]}
        if tool_name == "fail_tool":
            raise RuntimeError("This tool always fails")
        raise ValueError(f"Unknown tool: {tool_name}")


# ---------------------------------------------------------------------------
# Mock API response builders
# ---------------------------------------------------------------------------

def _make_text_response(text: str, stop_reason: str = "end_turn"):
    """Build a mock Anthropic API response with a text block."""
    content_block = MagicMock()
    content_block.type = "text"
    content_block.text = text

    response = MagicMock()
    response.stop_reason = stop_reason
    response.content = [content_block]
    return response


def _make_tool_use_response(tool_name: str, tool_input: dict, tool_id: str = "tool_abc123"):
    """Build a mock Anthropic API response requesting a tool call."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = tool_name
    tool_block.input = tool_input
    tool_block.id = tool_id

    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [tool_block]
    return response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBaseAgentLoop:

    def test_end_turn_on_first_response(self):
        """Agent returns immediately when model says end_turn with no tool calls."""
        agent = _EchoAgent()

        with patch.object(agent.client.messages, "create") as mock_create:
            mock_create.return_value = _make_text_response("The answer is 42.")
            result = agent.run("What is the answer?")

        assert result.agent_name == "EchoAgent"
        assert "42" in result.response
        assert len(result.tool_calls_made) == 0
        assert result.confidence > 0
        mock_create.assert_called_once()

    def test_single_tool_call_then_end_turn(self):
        """Agent calls one tool, gets result, then returns a final response."""
        agent = _EchoAgent()

        responses = [
            _make_tool_use_response("echo", {"message": "hello"}, tool_id="t1"),
            _make_text_response("The echo returned 'hello'. Done."),
        ]

        with patch.object(agent.client.messages, "create", side_effect=responses):
            result = agent.run("Echo hello please")

        assert len(result.tool_calls_made) == 1
        assert result.tool_calls_made[0]["tool"] == "echo"
        assert "hello" in result.response
        assert agent.tool_calls_received == [("echo", {"message": "hello"})]

    def test_tool_error_is_reported_to_model(self):
        """When a tool raises an exception, the error is returned as a tool_result."""
        agent = _EchoAgent()

        responses = [
            _make_tool_use_response("fail_tool", {}, tool_id="t_fail"),
            _make_text_response("The tool failed. I cannot complete the task."),
        ]

        with patch.object(agent.client.messages, "create", side_effect=responses) as mock_create:
            result = agent.run("Run the failing tool")

        # second call should include a tool_result with is_error=True
        second_call_messages = mock_create.call_args_list[1][1]["messages"]
        tool_results_msg = second_call_messages[-1]
        assert tool_results_msg["role"] == "user"
        tool_result_content = tool_results_msg["content"][0]
        assert tool_result_content.get("is_error") is True
        assert "fail" in tool_result_content["content"].lower()

    def test_max_tool_rounds_enforced(self):
        """Agent stops after max_tool_rounds even if model keeps requesting tools."""
        agent = _EchoAgent()
        agent.max_tool_rounds = 3

        # model always wants more tool calls
        always_tool = _make_tool_use_response("echo", {"message": "loop"})
        responses = [always_tool] * 10  # more than max_tool_rounds

        with patch.object(agent.client.messages, "create", side_effect=responses):
            result = agent.run("Keep calling tools forever")

        assert result.confidence == 0.0
        assert result.escalate is True
        assert agent.client.messages.create.call_count == agent.max_tool_rounds

    def test_multiple_tool_calls_in_one_round(self):
        """Agent can handle multiple tool_use blocks in a single response."""
        agent = _EchoAgent()

        # one response with two tool calls
        t1 = MagicMock()
        t1.type = "tool_use"
        t1.name = "echo"
        t1.input = {"message": "first"}
        t1.id = "tool_1"

        t2 = MagicMock()
        t2.type = "tool_use"
        t2.name = "echo"
        t2.input = {"message": "second"}
        t2.id = "tool_2"

        multi_tool_response = MagicMock()
        multi_tool_response.stop_reason = "tool_use"
        multi_tool_response.content = [t1, t2]

        responses = [multi_tool_response, _make_text_response("Got both echoes.")]

        with patch.object(agent.client.messages, "create", side_effect=responses):
            result = agent.run("Echo two things")

        assert len(result.tool_calls_made) == 2
        assert agent.tool_calls_received == [
            ("echo", {"message": "first"}),
            ("echo", {"message": "second"}),
        ]

    def test_context_is_appended_to_message(self):
        """Additional context dict should be serialised and appended to the user message."""
        agent = _EchoAgent()
        context = {"facility_id": "SYDE-01", "zone": "ZONE-A"}

        with patch.object(agent.client.messages, "create") as mock_create:
            mock_create.return_value = _make_text_response("OK")
            agent.run("Check something", context=context)

        call_kwargs = mock_create.call_args[1]
        user_message = call_kwargs["messages"][0]["content"]
        assert "SYDE-01" in user_message
        assert "ZONE-A" in user_message

    def test_memory_write_called_on_success(self):
        """Agent should write to memory when confidence is high enough."""
        mock_memory = MagicMock()
        agent = _EchoAgent(memory_client=mock_memory)

        with patch.object(agent.client.messages, "create") as mock_create:
            mock_create.return_value = _make_text_response("High confidence result.\nConfidence: 90%")
            result = agent.run("Test memory write")

        mock_memory.write_working.assert_called_once()
        call_kwargs = mock_memory.write_working.call_args[1]
        assert "EchoAgent" in call_kwargs["key"]

    def test_memory_not_written_on_low_confidence(self):
        """Agent should not write to memory when confidence is below threshold."""
        mock_memory = MagicMock()
        agent = _EchoAgent(memory_client=mock_memory)

        with patch.object(agent.client.messages, "create") as mock_create:
            mock_create.return_value = _make_text_response("Uncertain result.\nConfidence: 20%")
            result = agent.run("Test low confidence")

        mock_memory.write_working.assert_not_called()

    def test_escalation_parsed_from_response(self):
        """Agent should set escalate=True if response contains escalation signal."""
        agent = _EchoAgent()

        with patch.object(agent.client.messages, "create") as mock_create:
            mock_create.return_value = _make_text_response(
                "This situation is critical.\nConfidence: 85%\nEscalate: true"
            )
            result = agent.run("Check something serious")

        assert result.escalate is True

    def test_elapsed_ms_is_positive(self):
        """Elapsed time should be recorded and positive."""
        agent = _EchoAgent()
        with patch.object(agent.client.messages, "create") as mock_create:
            mock_create.return_value = _make_text_response("Done.")
            result = agent.run("Timing test")
        assert result.elapsed_ms > 0


class TestConfidenceParsing:

    def test_parse_percentage_confidence(self):
        agent = _EchoAgent()
        conf, _ = agent._parse_meta("Analysis complete.\nConfidence: 87%")
        assert abs(conf - 0.87) < 0.01

    def test_parse_decimal_confidence(self):
        agent = _EchoAgent()
        conf, _ = agent._parse_meta("Done.\nConfidence: 0.92")
        assert abs(conf - 0.92) < 0.01

    def test_no_confidence_defaults_to_1(self):
        agent = _EchoAgent()
        conf, _ = agent._parse_meta("No confidence marker here.")
        assert conf == 1.0

    def test_escalate_true_detected(self):
        agent = _EchoAgent()
        _, escalate = agent._parse_meta("Something is wrong.\nEscalate: true")
        assert escalate is True

    def test_escalate_false_not_triggered(self):
        agent = _EchoAgent()
        _, escalate = agent._parse_meta("All clear.\nEscalate: false")
        assert escalate is False
