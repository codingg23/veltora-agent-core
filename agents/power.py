"""
power.py

PowerAgent - monitors power draw, PUE, and energy efficiency.

Responsible for:
- Tracking PDU loads and UPS headroom
- Monitoring and explaining PUE trends
- Flagging racks approaching power limits
- Forecasting peak demand
- Estimating energy cost impact of changes
"""

from typing import Any
from .base import BaseAgent, SPECIALIST_MODEL
from tools.telemetry import TelemetryTools

SYSTEM_PROMPT = """You are the PowerAgent for a data centre operations platform.

Your job is to monitor power consumption, PUE, and energy efficiency. You have access
to tools that query real power telemetry from PDUs, UPS units, and facility meters.

When analysing power conditions:
- Always retrieve actual data before drawing conclusions
- Flag any PDU at >80% of rated capacity (N+1 headroom at risk)
- Flag any UPS at >70% load (approaching switchover threshold)
- PUE above 1.6 should be flagged as inefficient for a modern facility
- PUE above 2.0 is a serious concern - dig into what's driving it
- When PUE changes, check if IT load changed or if cooling overhead changed
- Correlate power spikes with thermal events if possible

When forecasting:
- Use historical patterns (day of week, time of day) to predict peaks
- Factor in any known planned workload changes

End your response with:
Confidence: [0-100]%
Escalate: [true/false]"""


class PowerAgent(BaseAgent):

    def __init__(self, telemetry_path: str, memory_client=None):
        super().__init__(name="PowerAgent", memory_client=memory_client, model=SPECIALIST_MODEL)
        self._tools = TelemetryTools(telemetry_path)

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    @property
    def tools(self) -> list[dict]:
        return [
            {
                "name": "query_pdu_load",
                "description": "Get PDU power draw for racks, rows, or facility. Returns kW and % of rated capacity.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "scope": {"type": "string", "description": "rack:R07-12, row:ROW-07, or facility"},
                        "start": {"type": "string"},
                        "end": {"type": "string"},
                        "aggregation": {"type": "string", "enum": ["avg", "max", "sum", "timeseries"], "default": "avg"},
                    },
                    "required": ["scope", "start", "end"],
                },
            },
            {
                "name": "query_pue",
                "description": "Get PUE metrics for the facility or a zone.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "facility": {"type": "string"},
                        "start": {"type": "string"},
                        "end": {"type": "string"},
                        "granularity": {"type": "string", "enum": ["hourly", "daily", "raw"], "default": "hourly"},
                    },
                    "required": ["facility", "start", "end"],
                },
            },
            {
                "name": "query_ups_status",
                "description": "Get UPS load percentage, runtime estimate, and bypass status.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ups_id": {"type": "string", "description": "UPS unit ID or 'all'"},
                    },
                    "required": ["ups_id"],
                },
            },
            {
                "name": "forecast_peak_demand",
                "description": "Forecast peak power demand for the next N hours based on historical patterns.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "scope": {"type": "string"},
                        "horizon_hours": {"type": "integer", "default": 24},
                    },
                    "required": ["scope"],
                },
            },
            {
                "name": "estimate_energy_cost",
                "description": "Estimate energy cost for a given load over a time period.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "load_kw": {"type": "number"},
                        "duration_hours": {"type": "number"},
                        "tariff_per_kwh": {"type": "number", "default": 0.12},
                    },
                    "required": ["load_kw", "duration_hours"],
                },
            },
            {
                "name": "resolve_time",
                "description": "Convert relative time to absolute timestamps.",
                "input_schema": {
                    "type": "object",
                    "properties": {"reference": {"type": "string"}},
                    "required": ["reference"],
                },
            },
        ]

    def execute_tool(self, tool_name: str, tool_input: dict) -> Any:
        dispatch = {
            "query_pdu_load": self._tools.query_power,
            "query_pue": self._tools.query_pue,
            "query_ups_status": self._tools.query_ups,
            "forecast_peak_demand": self._tools.forecast_power,
            "estimate_energy_cost": self._estimate_cost,
            "resolve_time": self._tools.resolve_time,
        }
        fn = dispatch.get(tool_name)
        if fn is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        return fn(**tool_input)

    def _estimate_cost(self, load_kw: float, duration_hours: float, tariff_per_kwh: float = 0.12) -> dict:
        kwh = load_kw * duration_hours
        cost = kwh * tariff_per_kwh
        return {
            "load_kw": load_kw,
            "duration_hours": duration_hours,
            "energy_kwh": round(kwh, 1),
            "cost_usd": round(cost, 2),
            "tariff_per_kwh": tariff_per_kwh,
        }
