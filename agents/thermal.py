"""
thermal.py

ThermalAgent - monitors temperature conditions across the facility.

Responsible for:
- Detecting ASHRAE limit breaches (A1: 15-32C inlet, A2: 10-35C)
- Identifying hot spots and recirculation issues
- Forecasting inlet temperatures under changing load
- Recommending cooling adjustments

The agent has access to real telemetry via DuckDB queries.
It does not make up numbers. If it can't retrieve data, it says so.
"""

import logging
from typing import Any
from .base import BaseAgent, SPECIALIST_MODEL
from tools.telemetry import TelemetryTools

logger = logging.getLogger(__name__)

ASHRAE_A1_MAX_INLET = 32.0   # C
ASHRAE_A2_MAX_INLET = 35.0
ASHRAE_A1_WARNING   = 27.0   # flag early before limit hit

SYSTEM_PROMPT = """You are the ThermalAgent for a data centre operations platform.

Your job is to monitor and analyse thermal conditions. You have access to tools that
query real temperature telemetry from sensors across the facility.

When analysing thermal conditions:
- Always retrieve actual data before drawing conclusions
- Flag any inlet temperature above 27C (approaching ASHRAE A1 limit of 32C)
- Flag any outlet temperature above 45C as potentially concerning
- Look for patterns: is it one rack, one row, or a zone-wide issue?
- Consider delta-T: a high delta-T (>15C) with moderate inlet is fine; the same delta-T
  with a high inlet (>27C) is not
- If you see a hot spot, check neighbouring racks too
- Suggest root causes: high density load, blanking panel missing, CRAC issue, containment breach

End your response with:
Confidence: [0-100]%
Escalate: [true/false]

Only escalate if inlet temps are above 30C or you see a runaway trend."""


class ThermalAgent(BaseAgent):

    def __init__(self, telemetry_path: str, memory_client=None):
        super().__init__(
            name="ThermalAgent",
            memory_client=memory_client,
            model=SPECIALIST_MODEL,
        )
        self._tools = TelemetryTools(telemetry_path)

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    @property
    def tools(self) -> list[dict]:
        return [
            {
                "name": "query_inlet_temps",
                "description": "Get inlet temperature readings for racks or rows. Returns avg, max, p95, and timeseries data.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "scope": {"type": "string", "description": "rack:R07-12, row:ROW-07, zone:ZONE-A, or facility"},
                        "start": {"type": "string", "description": "ISO 8601 start time"},
                        "end": {"type": "string", "description": "ISO 8601 end time"},
                        "aggregation": {"type": "string", "enum": ["avg", "max", "p95", "timeseries"], "default": "avg"},
                    },
                    "required": ["scope", "start", "end"],
                },
            },
            {
                "name": "query_outlet_temps",
                "description": "Get outlet temperature and delta-T for racks or rows.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "scope": {"type": "string"},
                        "start": {"type": "string"},
                        "end": {"type": "string"},
                    },
                    "required": ["scope", "start", "end"],
                },
            },
            {
                "name": "get_hot_spots",
                "description": "Identify racks currently above a temperature threshold. Returns ranked list.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "threshold_c": {"type": "number", "description": "Inlet temp threshold in Celsius", "default": 25.0},
                        "facility": {"type": "string", "default": "all"},
                    },
                    "required": [],
                },
            },
            {
                "name": "query_crac_status",
                "description": "Get CRAC unit status: load fraction, supply temp, return temp, COP.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "zone": {"type": "string", "description": "Zone ID or 'all'"},
                        "start": {"type": "string"},
                        "end": {"type": "string"},
                    },
                    "required": ["zone", "start", "end"],
                },
            },
            {
                "name": "forecast_inlet_temp",
                "description": "Forecast inlet temperature for next N hours given a load change.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "scope": {"type": "string"},
                        "horizon_hours": {"type": "integer", "default": 4},
                        "load_delta_kw": {"type": "number", "description": "Expected change in IT load in kW (positive = increase)"},
                    },
                    "required": ["scope", "load_delta_kw"],
                },
            },
            {
                "name": "resolve_time",
                "description": "Convert relative time to absolute ISO timestamps. Use for 'last hour', 'today', etc.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "reference": {"type": "string"},
                    },
                    "required": ["reference"],
                },
            },
        ]

    def execute_tool(self, tool_name: str, tool_input: dict) -> Any:
        dispatch = {
            "query_inlet_temps": self._tools.query_thermal,
            "query_outlet_temps": self._tools.query_thermal,
            "get_hot_spots": self._tools.get_hot_spots,
            "query_crac_status": self._tools.query_crac,
            "forecast_inlet_temp": self._tools.forecast_thermal,
            "resolve_time": self._tools.resolve_time,
        }
        fn = dispatch.get(tool_name)
        if fn is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        return fn(**tool_input)

    def check_facility(self, facility_id: str, time_window_hours: int = 1) -> dict:
        """
        Convenience method for the monitoring loop.
        Runs a broad thermal check across the whole facility.
        Returns a structured summary rather than a prose response.
        """
        task = (
            f"Run a full thermal health check for facility {facility_id}. "
            f"Check the last {time_window_hours} hour(s). "
            f"Identify any racks or zones above 25C inlet. "
            f"Check CRAC load fractions. "
            f"Flag anything that needs attention."
        )
        result = self.run(task)

        return {
            "agent": self.name,
            "facility": facility_id,
            "response": result.response,
            "confidence": result.confidence,
            "escalate": result.escalate,
            "tool_calls": len(result.tool_calls_made),
            "elapsed_ms": result.elapsed_ms,
        }
