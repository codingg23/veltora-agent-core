"""
capacity.py

CapacityAgent - tracks available space, power, and cooling capacity.

Responsible for:
- Maintaining a live view of available rack space and power budget
- Assessing impact of new workload provisioning requests
- Flagging when facility approaches critical capacity thresholds
- Recommending load placement to optimise space and cooling
"""

from typing import Any
from .base import BaseAgent, SPECIALIST_MODEL
from tools.telemetry import TelemetryTools

SYSTEM_PROMPT = """You are the CapacityAgent for a data centre operations platform.

Your job is to track and manage capacity across space, power, and cooling.

When assessing capacity:
- Space: available U-space per rack, per row, per zone
- Power: remaining kW budget per PDU, per row, per facility
- Cooling: CRAC headroom (>15% headroom is healthy, <5% is critical)

When evaluating a provisioning request:
- Check all three constraints (space, power, cooling)
- Recommend the best zone/row for placement based on available headroom
- Flag if any constraint will be exceeded
- Consider N+1 redundancy - available capacity should be "available minus one
  power circuit/cooling unit"

When capacity is tight:
- Identify underutilised racks that could absorb new load
- Check if load balancing across zones would help
- Flag if capacity planning needs to be updated

Be specific. "Row 7 has 120kW available on PDU-07B and 3 empty 2U slots
but only 8% CRAC headroom" is useful. "Capacity is limited" is not.

End with:
Confidence: [0-100]%
Escalate: [true/false]"""


class CapacityAgent(BaseAgent):

    def __init__(self, telemetry_path: str, memory_client=None):
        super().__init__(name="CapacityAgent", memory_client=memory_client, model=SPECIALIST_MODEL)
        self._tools = TelemetryTools(telemetry_path)

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    @property
    def tools(self) -> list[dict]:
        return [
            {
                "name": "get_capacity_map",
                "description": "Get current space, power, and cooling capacity for a scope.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "scope": {"type": "string", "description": "zone:ZONE-A, row:ROW-07, or facility"},
                        "include_reserved": {"type": "boolean", "default": True},
                    },
                    "required": ["scope"],
                },
            },
            {
                "name": "estimate_new_load_impact",
                "description": "Estimate the impact of adding new IT load at a given location.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "Target zone or row"},
                        "new_load_kw": {"type": "number"},
                        "rack_units": {"type": "integer", "description": "U-space required"},
                    },
                    "required": ["location", "new_load_kw", "rack_units"],
                },
            },
            {
                "name": "find_best_placement",
                "description": "Find the best location to place new workload given constraints.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "required_kw": {"type": "number"},
                        "required_u": {"type": "integer"},
                        "prefer_zone": {"type": "string", "description": "Optional preferred zone"},
                        "min_cooling_headroom_pct": {"type": "number", "default": 15.0},
                    },
                    "required": ["required_kw", "required_u"],
                },
            },
            {
                "name": "get_underutilised_racks",
                "description": "Find racks running at low utilisation that could absorb more load.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "scope": {"type": "string", "default": "facility"},
                        "threshold_pct": {"type": "number", "default": 30.0, "description": "Below this % utilisation"},
                    },
                    "required": [],
                },
            },
        ]

    def execute_tool(self, tool_name: str, tool_input: dict) -> Any:
        dispatch = {
            "get_capacity_map": self._tools.get_capacity,
            "estimate_new_load_impact": self._tools.estimate_load_impact,
            "find_best_placement": self._tools.find_placement,
            "get_underutilised_racks": self._tools.get_underutilised,
        }
        fn = dispatch.get(tool_name)
        if fn is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        return fn(**tool_input)
