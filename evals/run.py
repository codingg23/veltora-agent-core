"""
run.py

Evaluation harness for agent quality testing.

Runs a suite of eval cases against an agent and scores:
- Routing accuracy: did the agent call the expected tools?
- Fact accuracy: do required facts appear in the response?
- Forbidden pattern check: did the agent hallucinate or refuse incorrectly?
- Escalation accuracy: did the agent escalate when it should (or shouldn't)?

Usage:
    python -m evals.run --suite thermal_evals --telemetry ./data/eval_fixtures
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    case_id: str
    passed: bool
    routing_score: float
    fact_score: float
    escalation_correct: bool
    forbidden_triggered: list[str]
    response_preview: str
    tools_called: list[str]
    elapsed_ms: float
    notes: list[str] = field(default_factory=list)


def run_suite(
    suite_name: str,
    telemetry_path: str,
    model: Optional[str] = None,
    verbose: bool = False,
) -> list[EvalResult]:
    """Run all cases in a suite and return results."""
    from importlib import import_module

    suite_module = import_module(f"evals.{suite_name}")
    cases = suite_module.get_suite()

    from agents.thermal import ThermalAgent
    from agents.power import PowerAgent

    agent_map = {
        "thermal_evals": lambda: ThermalAgent(telemetry_path),
        "power_evals": lambda: PowerAgent(telemetry_path),
    }

    agent_factory = agent_map.get(suite_name)
    if agent_factory is None:
        raise ValueError(f"No agent mapping for suite: {suite_name}")

    agent = agent_factory()
    if model:
        agent.model = model

    results = []
    for case in cases:
        result = _run_case(agent, case, verbose=verbose)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {case.id}: routing={result.routing_score:.0%}, facts={result.fact_score:.0%}")
        if verbose and not result.passed:
            print(f"         Notes: {result.notes}")

    return results


def _run_case(agent, case, verbose: bool = False) -> EvalResult:
    from evals.thermal_evals import EvalCase

    t0 = time.time()
    try:
        agent_result = agent.run(case.prompt)
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        return EvalResult(
            case_id=case.id,
            passed=False,
            routing_score=0.0,
            fact_score=0.0,
            escalation_correct=False,
            forbidden_triggered=[],
            response_preview=f"EXCEPTION: {e}",
            tools_called=[],
            elapsed_ms=elapsed,
            notes=[f"Agent raised exception: {e}"],
        )

    elapsed = (time.time() - t0) * 1000
    tools_called = [t["tool"] for t in agent_result.tool_calls_made]
    response = agent_result.response.lower()
    notes = []

    # routing score: fraction of expected tools that were called
    if case.expected_tools:
        hits = sum(1 for t in case.expected_tools if t in tools_called)
        routing_score = hits / len(case.expected_tools)
    else:
        routing_score = 1.0

    # fact score: fraction of required facts that appear in response
    if case.required_facts:
        hits = sum(1 for f in case.required_facts if f.lower() in response)
        fact_score = hits / len(case.required_facts)
        missing = [f for f in case.required_facts if f.lower() not in response]
        if missing:
            notes.append(f"Missing facts: {missing}")
    else:
        fact_score = 1.0

    # forbidden patterns
    forbidden_triggered = [p for p in case.forbidden_patterns if p.lower() in response]
    if forbidden_triggered:
        notes.append(f"Forbidden patterns triggered: {forbidden_triggered}")

    # escalation
    escalation_correct = (agent_result.escalate == case.should_escalate)
    if not escalation_correct:
        notes.append(
            f"Escalation mismatch: expected={case.should_escalate}, got={agent_result.escalate}"
        )

    passed = (
        routing_score >= 0.75
        and fact_score >= 0.75
        and not forbidden_triggered
        and escalation_correct
    )

    return EvalResult(
        case_id=case.id,
        passed=passed,
        routing_score=routing_score,
        fact_score=fact_score,
        escalation_correct=escalation_correct,
        forbidden_triggered=forbidden_triggered,
        response_preview=agent_result.response[:200],
        tools_called=tools_called,
        elapsed_ms=round(elapsed, 1),
        notes=notes,
    )


def print_summary(results: list[EvalResult]):
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    routing = sum(r.routing_score for r in results) / total
    facts = sum(r.fact_score for r in results) / total
    escalation = sum(1 for r in results if r.escalation_correct) / total
    avg_ms = sum(r.elapsed_ms for r in results) / total

    print("\n" + "=" * 55)
    print(f"Results: {passed}/{total} passed ({passed/total:.0%})")
    print(f"  Routing accuracy:    {routing:.0%}")
    print(f"  Fact accuracy:       {facts:.0%}")
    print(f"  Escalation accuracy: {escalation:.0%}")
    print(f"  Avg latency:         {avg_ms:.0f}ms")

    failed = [r for r in results if not r.passed]
    if failed:
        print(f"\nFailed cases:")
        for r in failed:
            print(f"  {r.case_id}: {r.notes}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--telemetry", default="./data/eval_fixtures")
    parser.add_argument("--model", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"\nRunning eval suite: {args.suite}")
    print(f"Telemetry: {args.telemetry}")
    if args.model:
        print(f"Model: {args.model}")
    print()

    results = run_suite(args.suite, args.telemetry, args.model, args.verbose)
    print_summary(results)
