"""
coordinator.py

CoordinatorAgent - the top-level reasoning agent.

The coordinator sees everything:
- Results from all specialist agents
- Current facility state from working memory
- Historical incident patterns from episodic memory

Its job is to:
1. Route incoming tasks to the right specialist(s)
2. Synthesise cross-domain analysis (e.g. thermal + power correlation)
3. Make final decisions on incidents (escalate, auto-remediate, or dismiss)
4. Generate operator-facing summaries

Uses Opus because this is where the hard reasoning happens.
Specialists use Sonnet - fast and cheap for focused queries.
"""

import json
import logging
import time
from typing import Any, Optional
from .base import BaseAgent, COORDINATOR_MODEL, AgentResult
from .thermal import ThermalAgent
from .power import PowerAgent
from .capacity import CapacityAgent
from .incident import IncidentAgent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the CoordinatorAgent for a data centre AI operations platform.

You orchestrate a team of specialist agents:
- ThermalAgent: temperature, cooling, ASHRAE limits
- PowerAgent: PUE, PDU loads, UPS headroom, energy cost
- CapacityAgent: rack space, power budget, provisioning readiness
- IncidentAgent: alert triage, runbook lookup, escalation

You have access to their results and can delegate tasks to them.
You also have direct access to working memory (recent agent findings) and
episodic memory (past incidents and resolutions).

When you receive a task:
1. Decide which specialists are relevant
2. Delegate to them (they run in parallel where possible)
3. Look for cross-domain correlations in their results
4. Synthesise a final response that an operator can act on

Cross-domain reasoning examples:
- Thermal anomaly in ROW-07 + power spike in same zone = likely GPU workload burst
- High PUE + normal IT load = cooling system inefficiency, not workload driven
- Capacity request for +200kW + UPS at 80% = need to check UPS headroom before approving

When making decisions about incidents:
- AUTO-REMEDIATE: only if the action is low-risk and reversible (e.g. adjust CRAC setpoint)
- ESCALATE: if the situation requires human judgment or irreversible action
- MONITOR: if within acceptable range but trending toward a threshold
- DISMISS: if the alert is a known false positive (check episodic memory first)

Be concise. Operators are busy. Give them the answer + the key supporting data + the action.

End with:
Decision: [AUTO-REMEDIATE | ESCALATE | MONITOR | DISMISS]
Confidence: [0-100]%"""


class CoordinatorAgent(BaseAgent):

    def __init__(
        self,
        thermal_agent: ThermalAgent,
        power_agent: PowerAgent,
        capacity_agent: CapacityAgent,
        incident_agent: IncidentAgent,
        memory_client=None,
    ):
        super().__init__(
            name="CoordinatorAgent",
            memory_client=memory_client,
            model=COORDINATOR_MODEL,
            max_tool_rounds=8,
        )
        self.specialists = {
            "thermal": thermal_agent,
            "power": power_agent,
            "capacity": capacity_agent,
            "incident": incident_agent,
        }
        self._specialist_results: dict[str, AgentResult] = {}

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    @property
    def tools(self) -> list[dict]:
        return [
            {
                "name": "delegate_to_agent",
                "description": "Delegate a task to a specialist agent and get their analysis.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "enum": ["thermal", "power", "capacity", "incident"],
                            "description": "Which specialist agent to delegate to",
                        },
                        "task": {
                            "type": "string",
                            "description": "The task or question for the specialist",
                        },
                    },
                    "required": ["agent", "task"],
                },
            },
            {
                "name": "read_working_memory",
                "description": "Read recent agent findings and facility state from working memory.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "key_pattern": {
                            "type": "string",
                            "description": "Key pattern to search, e.g. 'ThermalAgent:*' or '*:last_result'",
                        },
                    },
                    "required": ["key_pattern"],
                },
            },
            {
                "name": "search_episodic_memory",
                "description": "Search past incidents and resolutions for similar situations.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language description of the situation",
                        },
                        "top_k": {"type": "integer", "default": 3},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "create_incident",
                "description": "Create a new incident record with severity, affected scope, and initial reasoning.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "severity": {"type": "string", "enum": ["P1", "P2", "P3", "P4"]},
                        "affected_scope": {"type": "string", "description": "e.g. 'ROW-07', 'ZONE-A', 'facility'"},
                        "description": {"type": "string"},
                        "recommended_action": {"type": "string"},
                    },
                    "required": ["title", "severity", "affected_scope", "description", "recommended_action"],
                },
            },
            {
                "name": "write_to_memory",
                "description": "Write coordinator decision or summary to shared memory for other agents to read.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "value": {"type": "object"},
                        "ttl_seconds": {"type": "integer", "default": 600},
                    },
                    "required": ["key", "value"],
                },
            },
        ]

    def execute_tool(self, tool_name: str, tool_input: dict) -> Any:
        if tool_name == "delegate_to_agent":
            return self._delegate(tool_input["agent"], tool_input["task"])

        if tool_name == "read_working_memory":
            if self.memory:
                return self.memory.read_pattern(tool_input["key_pattern"])
            return {"error": "memory not configured"}

        if tool_name == "search_episodic_memory":
            if self.memory:
                return self.memory.semantic_search(
                    tool_input["query"],
                    top_k=tool_input.get("top_k", 3),
                )
            return {"results": [], "note": "episodic memory not configured"}

        if tool_name == "create_incident":
            return self._create_incident(**tool_input)

        if tool_name == "write_to_memory":
            if self.memory:
                self.memory.write_working(
                    tool_input["key"],
                    tool_input["value"],
                    ttl=tool_input.get("ttl_seconds", 600),
                )
                return {"status": "written"}
            return {"status": "skipped - no memory"}

        raise ValueError(f"Unknown tool: {tool_name}")

    def _delegate(self, agent_name: str, task: str) -> dict:
        """Run a specialist agent and return their result."""
        agent = self.specialists.get(agent_name)
        if agent is None:
            return {"error": f"Unknown agent: {agent_name}"}

        logger.info(f"[Coordinator] Delegating to {agent_name}: {task[:60]}...")
        result = agent.run(task)
        self._specialist_results[agent_name] = result

        return {
            "agent": agent_name,
            "response": result.response,
            "confidence": result.confidence,
            "escalate": result.escalate,
            "tool_calls": len(result.tool_calls_made),
        }

    def _create_incident(self, title: str, severity: str, affected_scope: str,
                          description: str, recommended_action: str) -> dict:
        """Create an incident record."""
        incident_id = f"INC-{int(time.time())}"
        incident = {
            "id": incident_id,
            "title": title,
            "severity": severity,
            "affected_scope": affected_scope,
            "description": description,
            "recommended_action": recommended_action,
            "created_at": time.time(),
            "status": "open",
            "created_by": "CoordinatorAgent",
        }

        if self.memory:
            self.memory.write_working(f"incident:{incident_id}", incident, ttl=86400)

            # also write to episodic memory for future reference
            self.memory.write_episodic(
                text=f"{title}: {description}",
                metadata=incident,
            )

        logger.info(f"[Coordinator] Created incident {incident_id}: [{severity}] {title}")
        return incident

    def handle_alert(self, alert: dict, facility_id: str) -> AgentResult:
        """
        Entry point for handling an incoming alert from the monitoring system.

        Runs the full coordinator reasoning loop for one alert.
        """
        task = (
            f"An alert has been triggered for facility {facility_id}.\n\n"
            f"Alert details:\n{json.dumps(alert, indent=2)}\n\n"
            f"Investigate this alert. Delegate to the relevant specialist agents. "
            f"Check episodic memory for similar past incidents. "
            f"Decide whether to escalate, auto-remediate, monitor, or dismiss."
        )
        return self.run(task)

    def daily_summary(self, facility_id: str) -> AgentResult:
        """Generate a daily operational summary across all domains."""
        task = (
            f"Generate a daily operational summary for facility {facility_id}. "
            f"Delegate to all four specialist agents for their domain status. "
            f"Highlight anything trending in the wrong direction. "
            f"Compare against yesterday if data is available in memory. "
            f"Keep it concise - this is for an operator starting their shift."
        )
        return self.run(task)
# coordinator now uses Opus, specialists use Sonnet
