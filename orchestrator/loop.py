"""
loop.py

Main continuous monitoring loop.

Runs the agent network 24/7. Every N seconds:
1. Each specialist agent runs a health check for their domain
2. Results are written to working memory
3. If any agent flags an escalation, the coordinator investigates
4. The coordinator generates a daily summary at 06:00 local time

The loop is designed to be fault-tolerant:
- If one agent fails, the others keep running
- If the coordinator fails, monitoring continues without cross-domain analysis
- All exceptions are caught and logged - the loop never exits on its own

Usage:
    python -m orchestrator.loop --facility SYDE-01 --interval 60
"""

import argparse
import logging
import os
import signal
import time
from datetime import datetime
from typing import Optional

from agents.thermal import ThermalAgent
from agents.power import PowerAgent
from agents.capacity import CapacityAgent
from agents.incident import IncidentAgent
from agents.coordinator import CoordinatorAgent
from memory.working import WorkingMemory
from memory.episodic import EpisodicMemory

logger = logging.getLogger(__name__)


class AgentLoop:
    """
    Orchestrates continuous monitoring across all agents.
    """

    def __init__(
        self,
        facility_id: str,
        telemetry_path: str,
        redis_url: str = "redis://localhost:6379",
        pg_url: Optional[str] = None,
        interval_s: int = 60,
    ):
        self.facility_id = facility_id
        self.interval_s = interval_s
        self._running = False

        # memory
        host, port = self._parse_redis_url(redis_url)
        self.working_memory = WorkingMemory(host=host, port=port)
        self.episodic_memory = EpisodicMemory(pg_url) if pg_url else None

        # build shared memory client that agents use
        mem = _MemoryClient(self.working_memory, self.episodic_memory)

        # specialist agents
        self.thermal = ThermalAgent(telemetry_path, memory_client=mem)
        self.power = PowerAgent(telemetry_path, memory_client=mem)
        self.capacity = CapacityAgent(telemetry_path, memory_client=mem)
        self.incident = IncidentAgent(memory_client=mem)

        # coordinator
        self.coordinator = CoordinatorAgent(
            thermal_agent=self.thermal,
            power_agent=self.power,
            capacity_agent=self.capacity,
            incident_agent=self.incident,
            memory_client=mem,
        )

        self._last_daily_summary = None

        logger.info(f"AgentLoop initialised for facility {facility_id}, interval={interval_s}s")

    def _parse_redis_url(self, url: str) -> tuple[str, int]:
        # simple parser for redis://host:port
        url = url.replace("redis://", "")
        parts = url.split(":")
        host = parts[0]
        port = int(parts[1]) if len(parts) > 1 else 6379
        return host, port

    def run(self):
        """Main loop. Runs until SIGINT/SIGTERM."""
        self._running = True
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        logger.info(f"Starting agent monitoring loop for {self.facility_id}")

        while self._running:
            loop_start = time.time()

            try:
                self._run_monitoring_cycle()
            except Exception as e:
                logger.exception(f"Monitoring cycle failed: {e}")

            # check if we should run daily summary
            try:
                self._maybe_run_daily_summary()
            except Exception as e:
                logger.exception(f"Daily summary failed: {e}")

            elapsed = time.time() - loop_start
            sleep_time = max(0, self.interval_s - elapsed)
            if sleep_time > 0:
                logger.debug(f"Cycle done in {elapsed:.1f}s, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)

    def _run_monitoring_cycle(self):
        """Run one monitoring cycle across all domains."""
        escalations = []

        agents_to_run = [
            ("thermal", lambda: self.thermal.check_facility(self.facility_id, time_window_hours=1)),
            ("power", lambda: self._run_power_check()),
            ("capacity", lambda: self._run_capacity_check()),
        ]

        for agent_name, fn in agents_to_run:
            try:
                result = fn()
                self.working_memory.agent_heartbeat(agent_name)

                if result.get("escalate"):
                    escalations.append((agent_name, result))
                    logger.warning(f"[{agent_name}] Escalation flagged")

            except Exception as e:
                logger.error(f"Agent {agent_name} failed in monitoring cycle: {e}")

        # if any specialist flagged escalation, bring in coordinator
        if escalations:
            self._coordinate_escalations(escalations)

    def _run_power_check(self) -> dict:
        result = self.power.run(
            f"Quick power health check for facility {self.facility_id}. "
            f"Check PUE and flag any PDUs above 80% load. Last 1 hour."
        )
        return {
            "agent": "power",
            "response": result.response,
            "escalate": result.escalate,
            "confidence": result.confidence,
        }

    def _run_capacity_check(self) -> dict:
        result = self.capacity.run(
            f"Quick capacity check for facility {self.facility_id}. "
            f"Flag any zones with less than 10% cooling or power headroom."
        )
        return {
            "agent": "capacity",
            "response": result.response,
            "escalate": result.escalate,
            "confidence": result.confidence,
        }

    def _coordinate_escalations(self, escalations: list):
        """Hand escalations to the coordinator for cross-domain analysis."""
        escalation_summary = "\n\n".join([
            f"[{name}]: {result.get('response', '')[:300]}"
            for name, result in escalations
        ])

        task = (
            f"Multiple specialist agents have flagged escalations for facility {self.facility_id}:\n\n"
            f"{escalation_summary}\n\n"
            f"Investigate across domains, check for correlations, and decide on the appropriate response."
        )

        logger.info(f"Coordinator handling {len(escalations)} escalation(s)")
        result = self.coordinator.run(task)
        logger.info(f"Coordinator decision: {result.response[:200]}")

    def _maybe_run_daily_summary(self):
        """Run daily summary at 06:00 if not already done today."""
        now = datetime.now()
        today = now.date()

        if now.hour != 6:
            return
        if self._last_daily_summary == today:
            return

        logger.info("Running daily summary")
        result = self.coordinator.daily_summary(self.facility_id)
        self.working_memory.write(
            "coordinator:daily_summary",
            {"date": str(today), "summary": result.response, "ts": time.time()},
            ttl=86400,
        )
        self._last_daily_summary = today
        logger.info("Daily summary complete")

    def _shutdown(self, signum, frame):
        logger.info("Shutdown signal received")
        self._running = False


class _MemoryClient:
    """Adapter that gives agents a unified interface to both memory layers."""

    def __init__(self, working: WorkingMemory, episodic: Optional[EpisodicMemory]):
        self._w = working
        self._e = episodic

    def write_working(self, key: str, value, ttl: int = 300):
        self._w.write(key, value, ttl=ttl)

    def read_pattern(self, pattern: str) -> dict:
        return self._w.read_pattern(pattern)

    def write_episodic(self, text: str, metadata: dict) -> str:
        if self._e:
            return self._e.write(text, metadata)
        return ""

    def semantic_search(self, query: str, top_k: int = 5) -> dict:
        if self._e:
            return self._e.search_as_dict(query, top_k=top_k)
        return {"results": [], "note": "episodic memory not configured"}


def main():
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(description="Veltora agent monitoring loop")
    parser.add_argument("--facility", required=True, help="Facility ID, e.g. SYDE-01")
    parser.add_argument("--interval", type=int, default=60, help="Monitoring interval in seconds")
    parser.add_argument("--telemetry", default=os.getenv("TELEMETRY_PATH", "./data"))
    parser.add_argument("--redis", default=os.getenv("REDIS_URL", "redis://localhost:6379"))
    parser.add_argument("--pgvector", default=os.getenv("DATABASE_URL"))
    args = parser.parse_args()

    loop = AgentLoop(
        facility_id=args.facility,
        telemetry_path=args.telemetry,
        redis_url=args.redis,
        pg_url=args.pgvector,
        interval_s=args.interval,
    )
    loop.run()


if __name__ == "__main__":
    main()
# daily summary now runs at 06:00 facility local time
