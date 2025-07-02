"""
thermal_evals.py

Evaluation suite for ThermalAgent.

Each eval case has:
- A prompt (what the agent receives)
- Expected routing (which tools should be called)
- Required facts (strings that must appear in the response)
- Forbidden facts (things the agent should NOT say without data)

Run with:
    python -m evals.run --suite thermal_evals --model claude-sonnet-4-6
"""

from dataclasses import dataclass, field


@dataclass
class EvalCase:
    id: str
    prompt: str
    required_facts: list[str] = field(default_factory=list)
    forbidden_patterns: list[str] = field(default_factory=list)
    expected_tools: list[str] = field(default_factory=list)
    should_escalate: bool = False
    description: str = ""


THERMAL_EVALS = [
    EvalCase(
        id="thermal-01",
        description="Basic inlet temp query - should retrieve data and report accurately",
        prompt="What is the current inlet temperature in ROW-07?",
        expected_tools=["query_inlet_temps"],
        required_facts=["ROW-07", "°C"],
        forbidden_patterns=["I estimate", "approximately", "probably around"],
    ),
    EvalCase(
        id="thermal-02",
        description="Hot spot detection - should identify the highest-temp rack",
        prompt="Are there any hot spots in the facility right now? I'm worried about Zone A.",
        expected_tools=["get_hot_spots", "query_inlet_temps"],
        required_facts=["°C"],
        forbidden_patterns=["I don't have access", "cannot retrieve"],
    ),
    EvalCase(
        id="thermal-03",
        description="ASHRAE limit breach - must escalate if inlet > 30C",
        prompt="ROW-03 rack R03-08 inlet temp is showing 31.5C. What should I do?",
        expected_tools=["query_inlet_temps", "query_crac_status"],
        required_facts=["ASHRAE", "31"],
        should_escalate=True,
    ),
    EvalCase(
        id="thermal-04",
        description="Relative time handling - must call resolve_time before querying",
        prompt="How did inlet temps in Zone B look this afternoon?",
        expected_tools=["resolve_time", "query_inlet_temps"],
        required_facts=["°C"],
    ),
    EvalCase(
        id="thermal-05",
        description="CRAC status check - should include load fraction and COP",
        prompt="Is the cooling system keeping up with the current load?",
        expected_tools=["query_crac_status", "query_inlet_temps"],
        required_facts=["%"],
        forbidden_patterns=["I cannot determine", "I don't know"],
    ),
    EvalCase(
        id="thermal-06",
        description="Forecast under load change - should give numerical forecast",
        prompt="We're planning to add 80kW to ROW-05 tomorrow. Will temperatures stay within limits?",
        expected_tools=["forecast_inlet_temp"],
        required_facts=["°C", "kW"],
        forbidden_patterns=["I cannot predict", "impossible to say"],
    ),
    EvalCase(
        id="thermal-07",
        description="No data available scenario - should say so clearly, not make up numbers",
        prompt="What was the inlet temp in RACK-R99-99 last year?",
        expected_tools=["query_inlet_temps"],
        forbidden_patterns=["was approximately", "was around", "was roughly"],
        required_facts=["not found", "no data", "unavailable", "error"],
    ),
    EvalCase(
        id="thermal-08",
        description="Multi-rack comparison - should compare and rank",
        prompt="Which row had the worst thermal performance last week?",
        expected_tools=["resolve_time", "query_inlet_temps"],
        required_facts=["°C"],
    ),
    EvalCase(
        id="thermal-09",
        description="Delta-T analysis - should consider both inlet and delta-T",
        prompt="R07-12 has a delta-T of 18C. Is that a problem?",
        expected_tools=["query_inlet_temps", "query_outlet_temps"],
        required_facts=["inlet", "delta"],
        forbidden_patterns=["yes, that's a problem", "no, that's fine"],  # must check actual data
    ),
    EvalCase(
        id="thermal-10",
        description="Trend analysis - should identify if temps are rising or stable",
        prompt="Has ROW-07 been getting hotter over the past 6 hours?",
        expected_tools=["resolve_time", "query_inlet_temps"],
        required_facts=["°C"],
    ),
]


def get_suite() -> list[EvalCase]:
    return THERMAL_EVALS
