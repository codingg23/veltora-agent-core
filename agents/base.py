"""
base.py

Base class for all Veltora agents.

Every agent follows the same pattern:
  1. Receive a task (natural language or structured)
  2. Run the agentic loop: call tools until enough information is gathered
  3. Write key findings to shared memory
  4. Return a structured result

The base class handles the loop, tool dispatch, memory writes, and logging.
Subclasses define their system prompt, tools, and domain-specific logic.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import anthropic

logger = logging.getLogger(__name__)

# All agents use the same model. Coordinator uses Opus for cross-domain reasoning,
# specialists use Sonnet (faster, cheaper, sufficient for focused domains).
SPECIALIST_MODEL = "claude-sonnet-4-6"
COORDINATOR_MODEL = "claude-opus-4-6"


@dataclass
class AgentResult:
    agent_name: str
    task: str
    response: str
    tool_calls_made: list[dict] = field(default_factory=list)
    memory_writes: list[str] = field(default_factory=list)
    confidence: float = 1.0   # 0-1, agent's self-assessed confidence
    escalate: bool = False     # True = coordinator should review this
    elapsed_ms: float = 0.0


@dataclass
class ToolResult:
    tool_use_id: str
    content: Any
    error: Optional[str] = None

    def to_api_format(self) -> dict:
        if self.error:
            return {
                "type": "tool_result",
                "tool_use_id": self.tool_use_id,
                "content": f"Error: {self.error}",
                "is_error": True,
            }
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": json.dumps(self.content) if not isinstance(self.content, str) else self.content,
        }


class BaseAgent(ABC):
    """
    Base agent class. All specialist agents and the coordinator extend this.

    Subclasses must implement:
    - system_prompt: str property
    - tools: list property (Anthropic tool definitions)
    - execute_tool(name, input) -> any
    """

    def __init__(
        self,
        name: str,
        memory_client=None,
        model: str = SPECIALIST_MODEL,
        max_tool_rounds: int = 6,
    ):
        self.name = name
        self.memory = memory_client
        self.model = model
        self.max_tool_rounds = max_tool_rounds
        self.client = anthropic.Anthropic()
        self._call_count = 0

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        pass

    @property
    @abstractmethod
    def tools(self) -> list[dict]:
        pass

    @abstractmethod
    def execute_tool(self, tool_name: str, tool_input: dict) -> Any:
        pass

    def run(self, task: str, context: Optional[dict] = None) -> AgentResult:
        """
        Run the agent on a task. Handles the full agentic loop.

        context: optional dict with additional context (e.g. results from other agents,
                 relevant memory entries, current facility state).
        """
        t0 = time.time()
        self._call_count += 1

        user_message = task
        if context:
            ctx_str = json.dumps(context, indent=2)
            user_message = f"{task}\n\nAdditional context:\n{ctx_str}"

        messages = [{"role": "user", "content": user_message}]
        tool_calls_log = []

        logger.info(f"[{self.name}] Starting task: {task[:80]}...")

        for round_num in range(self.max_tool_rounds):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    system=self.system_prompt,
                    tools=self.tools if self.tools else anthropic.NOT_GIVEN,
                    messages=messages,
                )
            except anthropic.APIError as e:
                logger.error(f"[{self.name}] API error on round {round_num + 1}: {e}")
                raise

            logger.debug(f"[{self.name}] Round {round_num + 1}: stop_reason={response.stop_reason}")

            if response.stop_reason == "end_turn":
                text_blocks = [b.text for b in response.content if hasattr(b, "text")]
                final_response = "\n".join(text_blocks)

                # parse confidence and escalation from response if present
                confidence, escalate = self._parse_meta(final_response)

                elapsed = (time.time() - t0) * 1000
                result = AgentResult(
                    agent_name=self.name,
                    task=task,
                    response=final_response,
                    tool_calls_made=tool_calls_log,
                    confidence=confidence,
                    escalate=escalate,
                    elapsed_ms=round(elapsed, 1),
                )

                # write summary to memory if available
                if self.memory and confidence > 0.5:
                    self._write_to_memory(task, final_response)
                    result.memory_writes.append("summary")

                logger.info(f"[{self.name}] Done in {elapsed:.0f}ms, {len(tool_calls_log)} tool calls")
                return result

            if response.stop_reason != "tool_use":
                logger.warning(f"[{self.name}] Unexpected stop_reason: {response.stop_reason}")
                break

            # execute tool calls
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                logger.info(f"[{self.name}] Tool call: {block.name}({json.dumps(block.input)[:100]})")
                tool_calls_log.append({"tool": block.name, "input": block.input})

                try:
                    result_data = self.execute_tool(block.name, block.input)
                    tr = ToolResult(tool_use_id=block.id, content=result_data)
                except Exception as e:
                    logger.warning(f"[{self.name}] Tool {block.name} failed: {e}")
                    tr = ToolResult(tool_use_id=block.id, content=None, error=str(e))

                tool_results.append(tr.to_api_format())

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        # fell through - return whatever we have
        elapsed = (time.time() - t0) * 1000
        return AgentResult(
            agent_name=self.name,
            task=task,
            response="Could not complete reasoning within tool round limit.",
            tool_calls_made=tool_calls_log,
            confidence=0.0,
            escalate=True,
            elapsed_ms=round(elapsed, 1),
        )

    def _parse_meta(self, response_text: str) -> tuple[float, bool]:
        """
        Extract confidence and escalation flag from response text.
        Agents are instructed to include these in structured form at the end.
        """
        confidence = 1.0
        escalate = False

        for line in response_text.lower().split("\n"):
            if line.startswith("confidence:"):
                try:
                    val = line.split(":")[1].strip().rstrip("%")
                    confidence = float(val) / 100 if float(val) > 1 else float(val)
                except (ValueError, IndexError):
                    pass
            if "escalate: true" in line or "needs review" in line:
                escalate = True

        return confidence, escalate

    def _write_to_memory(self, task: str, response: str):
        """Write a summary of this agent run to shared memory."""
        try:
            self.memory.write_working(
                key=f"{self.name}:last_result",
                value={"task": task[:200], "summary": response[:500], "ts": time.time()},
                ttl=300,  # 5 minute TTL in working memory
            )
        except Exception as e:
            logger.debug(f"[{self.name}] Memory write failed: {e}")
