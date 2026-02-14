"""
incident.py

IncidentAgent - alert triage and incident management.

Responsible for:
- Triaging incoming alerts (is this real? how urgent?)
- Searching runbooks for relevant remediation steps
- Determining if an incident matches a known pattern (false positive, known issue)
- Deciding severity and escalation path
- Writing structured incident records
"""

from typing import Any
from .base import BaseAgent, SPECIALIST_MODEL
from tools.runbooks import RunbookTools

SYSTEM_PROMPT = """You are the IncidentAgent for a data centre operations platform.

Your job is to triage alerts and manage incidents. You are the first to assess whether
an alert is real, how severe it is, and what to do next.

When triaging an alert:
1. Check if it matches a known false positive pattern (search runbooks + episodic memory)
2. Assess severity based on scope, trend, and proximity to thresholds
3. Find the relevant runbook if one exists
4. Recommend a specific action: auto-remediate, page on-call, or monitor

Severity guidelines:
- P1: Active or imminent customer impact, requires immediate action (<15 min response)
- P2: Degraded state, no immediate customer impact but trending toward P1 (1h response)
- P3: Anomaly detected, within acceptable range but needs monitoring (4h response)
- P4: Informational, no action required (next business day)

When searching runbooks, match on:
- Alert type (thermal, power, cooling, hardware)
- Affected component type (CRAC, PDU, UPS, rack)
- Symptom description

Be decisive. Operators don't want "it depends" - they want a clear recommendation
with the supporting reasoning so they can override if needed.

End with:
Severity: [P1/P2/P3/P4]
Confidence: [0-100]%
Escalate: [true/false]"""


class IncidentAgent(BaseAgent):

    def __init__(self, memory_client=None):
        super().__init__(name="IncidentAgent", memory_client=memory_client, model=SPECIALIST_MODEL)
        self._runbooks = RunbookTools()

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    @property
    def tools(self) -> list[dict]:
        return [
            {
                "name": "search_runbooks",
                "description": "Search the runbook library for procedures matching an alert type or symptom.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Alert type or symptom description"},
                        "component": {"type": "string", "description": "Component type: CRAC, PDU, UPS, rack, chiller"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_open_incidents",
                "description": "Get currently open incidents, optionally filtered by scope or severity.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "scope": {"type": "string", "description": "Facility, zone, or row to filter by"},
                        "min_severity": {"type": "string", "enum": ["P1", "P2", "P3", "P4"], "default": "P3"},
                    },
                    "required": [],
                },
            },
            {
                "name": "search_past_incidents",
                "description": "Search historical incidents for similar situations and their resolutions.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "days_back": {"type": "integer", "default": 90},
                        "top_k": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "check_false_positive_patterns",
                "description": "Check if an alert matches known false positive patterns for this facility.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "alert_type": {"type": "string"},
                        "scope": {"type": "string"},
                        "value": {"type": "number", "description": "The metric value that triggered the alert"},
                    },
                    "required": ["alert_type", "scope"],
                },
            },
            {
                "name": "page_oncall",
                "description": "Page the on-call engineer. Use only for P1/P2 incidents.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "incident_id": {"type": "string"},
                        "message": {"type": "string", "description": "Short message for the page (max 160 chars)"},
                        "severity": {"type": "string", "enum": ["P1", "P2"]},
                    },
                    "required": ["incident_id", "message", "severity"],
                },
            },
        ]

    def execute_tool(self, tool_name: str, tool_input: dict) -> Any:
        if tool_name == "search_runbooks":
            return self._runbooks.search(**tool_input)
        if tool_name == "get_open_incidents":
            return self._get_open_incidents(**tool_input)
        if tool_name == "search_past_incidents":
            if self.memory:
                return self.memory.semantic_search(tool_input["query"], tool_input.get("top_k", 5))
            return {"results": [], "note": "episodic memory not available"}
        if tool_name == "check_false_positive_patterns":
            return self._runbooks.check_false_positive(**tool_input)
        if tool_name == "page_oncall":
            return self._page_oncall(**tool_input)
        raise ValueError(f"Unknown tool: {tool_name}")

    def _get_open_incidents(self, scope: str = None, min_severity: str = "P3") -> dict:
        if not self.memory:
            return {"incidents": [], "note": "memory not available"}
        pattern = "incident:*"
        raw = self.memory.read_pattern(pattern)
        severity_order = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
        min_sev_num = severity_order.get(min_severity, 3)
        incidents = []
        for item in (raw or {}).values():
            if isinstance(item, dict) and item.get("status") == "open":
                sev = item.get("severity", "P4")
                if severity_order.get(sev, 4) <= min_sev_num:
                    if scope is None or scope in item.get("affected_scope", ""):
                        incidents.append(item)
        incidents.sort(key=lambda x: severity_order.get(x.get("severity", "P4"), 4))
        return {"incidents": incidents, "count": len(incidents)}

    def _page_oncall(self, incident_id: str, message: str, severity: str) -> dict:
        # in production this would call PagerDuty / OpsGenie API
        import logging
        logging.getLogger(__name__).warning(
            f"ONCALL PAGE [{severity}] {incident_id}: {message}"
        )
        return {
            "status": "paged",
            "incident_id": incident_id,
            "severity": severity,
            "note": "On-call engineer notified via PagerDuty",
        }
# incident agent now checks episodic memory before paging on-call
