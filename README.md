# veltora-agent-core

Autonomous multi-agent system for data centre operations.

The core idea: instead of a single LLM answering questions, run a network of specialised
agents that monitor, reason, and act continuously. Each agent owns a narrow domain
(thermal, power, capacity, incidents) and communicates through a shared memory layer.
A coordinator agent routes tasks and synthesises cross-domain decisions.

This is what runs behind the Veltora platform in production. The agent loop runs 24/7,
not just when an operator sends a query.

## Architecture

```
                        ┌─────────────────────┐
                        │   Coordinator Agent  │
                        │  (routes + decides)  │
                        └────────┬────────────┘
                                 │
           ┌─────────────────────┼──────────────────────┐
           │                     │                      │
    ┌──────▼──────┐    ┌─────────▼──────┐    ┌─────────▼──────┐
    │  Thermal    │    │    Power &     │    │   Capacity &   │
    │   Agent     │    │   PUE Agent    │    │  Incident Agent│
    └──────┬──────┘    └─────────┬──────┘    └─────────┬──────┘
           │                     │                      │
           └─────────────────────▼──────────────────────┘
                        ┌────────────────┐
                        │  Shared Memory │
                        │  (Redis + vec) │
                        └────────────────┘
```

Each agent has:
- A system prompt scoped to its domain
- A set of tools that query real telemetry (DuckDB over Parquet)
- Read/write access to shared memory
- The ability to escalate to the coordinator or raise incidents

The coordinator sees the full picture and makes cross-domain decisions - e.g. "thermal
anomaly in ROW-07 + power spike in the same zone = likely GPU workload burst, check
capacity plan before raising a cooling alarm."

## Agents

| Agent | Domain | Key tools |
|-------|--------|-----------|
| `ThermalAgent` | Temperature, ASHRAE limits, cooling | query_thermal, raycast_hot_spots, predict_inlet_temp |
| `PowerAgent` | PUE, PDU loads, UPS headroom | query_power, forecast_peak, check_ups_margin |
| `CapacityAgent` | Rack space, power budget, provisioning | get_capacity_map, estimate_new_load_impact |
| `IncidentAgent` | Alert triage, escalation, runbooks | get_open_incidents, search_runbooks, page_oncall |
| `CoordinatorAgent` | Cross-domain reasoning, final decisions | all of the above, plus delegate_to_agent |

## Memory

Two memory layers:
- **Working memory** (Redis): short-lived context, agent state, in-flight tasks
- **Episodic memory** (pgvector): past incidents + resolutions, indexed for semantic search

When an agent sees a new anomaly, it first searches episodic memory for similar past
events. If there's a match, it surfaces the resolution path. If not, it reasons from
scratch and writes the result back to memory.

## Running locally

```bash
pip install -r requirements.txt

# start Redis for working memory
docker run -d -p 6379:6379 redis:alpine

# set env vars
export ANTHROPIC_API_KEY=...
export DATABASE_URL=postgresql://...   # for pgvector
export TELEMETRY_PATH=./data/          # path to parquet files

# run the agent loop
python -m orchestrator.loop --facility SYDE-01 --interval 60

# or run the FastAPI server (query agents on demand)
uvicorn api.server:app --reload
```

## API

```
POST /query          - ask any question, coordinator routes to right agent
POST /agents/{name}  - talk directly to a specific agent
GET  /incidents      - current open incidents with agent reasoning
GET  /memory/search  - semantic search over episodic memory
GET  /health         - agent status + last heartbeat times
```

## Evals

Agent quality is measured against a set of labelled scenarios in `evals/`.
Each eval has a prompt, expected agent routing, and expected key facts in the response.

```bash
python -m evals.run --suite thermal_evals --model claude-opus-4-6
```

Current scores on the thermal eval suite: routing accuracy 94%, fact accuracy 91%.

## File layout

```
agents/          - individual agent implementations
  thermal.py     - ThermalAgent
  power.py       - PowerAgent
  capacity.py    - CapacityAgent
  incident.py    - IncidentAgent
  coordinator.py - CoordinatorAgent
tools/           - tool functions called by agents
  telemetry.py   - DuckDB queries over Parquet
  memory.py      - read/write to Redis + pgvector
  runbooks.py    - structured runbook lookup
orchestrator/    - agent loop, task routing, scheduling
  loop.py        - main continuous monitoring loop
  router.py      - decides which agent handles a task
memory/          - memory layer implementations
  working.py     - Redis working memory
  episodic.py    - pgvector episodic memory
api/             - FastAPI server for on-demand queries
evals/           - evaluation harness + labelled test cases
tests/           - unit + integration tests
```

## Why multi-agent instead of one big agent?

Two reasons:
1. **Context window**: a single agent trying to monitor thermal, power, capacity, and
   incidents simultaneously needs too much context. Specialised agents stay focused.
2. **Parallel execution**: thermal and power checks can run concurrently. A single
   agent is sequential. With 4 specialist agents + a coordinator, we check everything
   in one round-trip instead of four.

The tradeoff is coordination complexity. The coordinator prompt took the most iteration
to get right - it needs to know when to trust a specialist vs when to investigate further.

Current eval results (2026-03-10): routing accuracy 94%, fact accuracy 91%.
