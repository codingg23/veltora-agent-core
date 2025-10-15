"""
server.py

FastAPI server for on-demand agent queries.

Endpoints:
  POST /query              - ask anything, coordinator routes to right agent(s)
  POST /agents/{name}      - talk directly to a specialist agent
  GET  /incidents          - current open incidents with agent reasoning
  GET  /agents/status      - heartbeat status of all running agents
  GET  /memory/search      - semantic search over episodic memory
  GET  /health             - overall system health

All responses include the agent's reasoning trace (which tools it called and why)
so operators can audit every decision. This is important for trust - you should
never have to take the agent's word for it.

Authentication: Bearer token via VELTORA_API_KEY env var.
Rate limiting: 60 requests/min per token (in-memory, resets on restart).
"""

import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000, description="Natural language query")
    facility_id: str = Field(default="default", description="Facility to query against")
    include_trace: bool = Field(default=True, description="Include tool call trace in response")
    stream: bool = Field(default=False, description="Stream the response via SSE")


class AgentQueryRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=4000)
    context: Optional[dict] = Field(default=None, description="Optional additional context")


class QueryResponse(BaseModel):
    query: str
    agent: str
    response: str
    confidence: float
    escalated: bool
    tool_calls: list[dict]
    elapsed_ms: float
    facility_id: str


class IncidentResponse(BaseModel):
    incidents: list[dict]
    count: int
    facility_id: str


class MemorySearchResponse(BaseModel):
    query: str
    results: list[dict]
    count: int


# ---------------------------------------------------------------------------
# Rate limiting (simple token bucket per API key)
# ---------------------------------------------------------------------------

_rate_buckets: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT = 60  # requests per minute


def _check_rate_limit(api_key: str):
    now = time.time()
    window_start = now - 60
    bucket = _rate_buckets[api_key]
    # remove old entries
    _rate_buckets[api_key] = [t for t in bucket if t > window_start]
    if len(_rate_buckets[api_key]) >= RATE_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({RATE_LIMIT} requests/min)",
        )
    _rate_buckets[api_key].append(now)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

VALID_API_KEYS = set(filter(None, [os.getenv("VELTORA_API_KEY"), os.getenv("VELTORA_API_KEY_2")]))


def get_api_key(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Bearer token")
    token = auth[len("Bearer "):]
    if VALID_API_KEYS and token not in VALID_API_KEYS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    _check_rate_limit(token)
    return token


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

_agents: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise agents on startup, clean up on shutdown."""
    from agents.thermal import ThermalAgent
    from agents.power import PowerAgent
    from agents.capacity import CapacityAgent
    from agents.incident import IncidentAgent
    from agents.coordinator import CoordinatorAgent
    from memory.working import WorkingMemory
    from memory.episodic import EpisodicMemory

    telemetry_path = os.getenv("TELEMETRY_PATH", "./data")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    pg_url = os.getenv("DATABASE_URL")

    host, port_str = redis_url.replace("redis://", "").split(":")
    working_mem = WorkingMemory(host=host, port=int(port_str))
    episodic_mem = EpisodicMemory(pg_url) if pg_url else None

    class _Mem:
        def write_working(self, k, v, ttl=300): working_mem.write(k, v, ttl=ttl)
        def read_pattern(self, p): return working_mem.read_pattern(p)
        def write_episodic(self, text, metadata): return episodic_mem.write(text, metadata) if episodic_mem else ""
        def semantic_search(self, q, top_k=5): return episodic_mem.search_as_dict(q, top_k) if episodic_mem else {"results": []}

    mem = _Mem()
    thermal = ThermalAgent(telemetry_path, memory_client=mem)
    power = PowerAgent(telemetry_path, memory_client=mem)
    capacity = CapacityAgent(telemetry_path, memory_client=mem)
    incident = IncidentAgent(memory_client=mem)
    coordinator = CoordinatorAgent(thermal, power, capacity, incident, memory_client=mem)

    _agents.update({
        "thermal": thermal, "power": power,
        "capacity": capacity, "incident": incident,
        "coordinator": coordinator, "working_memory": working_mem,
        "episodic_memory": episodic_mem,
    })
    logger.info("All agents initialised")
    yield
    logger.info("Shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Veltora Agent API",
    description="Multi-agent system for data centre operations intelligence",
    version="0.4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """System health - agent status, memory connectivity."""
    working_mem: WorkingMemory = _agents.get("working_memory")
    agent_status = working_mem.get_agent_status() if working_mem else {}
    return {
        "status": "ok",
        "agents": agent_status,
        "memory": {
            "working": working_mem is not None,
            "episodic": _agents.get("episodic_memory") is not None,
        },
        "ts": time.time(),
    }


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, api_key: str = Depends(get_api_key)):
    """
    Send a natural language query. The coordinator routes it to the right agent(s).

    Examples:
    - "What is the inlet temperature in ROW-07 right now?"
    - "Are we approaching any capacity limits in Zone A?"
    - "What caused the PUE spike at 3pm yesterday?"
    - "Can we add 150kW of new load to the west hall?"
    """
    coordinator = _agents.get("coordinator")
    if not coordinator:
        raise HTTPException(status_code=503, detail="Agents not initialised")

    try:
        result = coordinator.run(
            task=req.query,
            context={"facility_id": req.facility_id},
        )
    except Exception as e:
        logger.exception(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    return QueryResponse(
        query=req.query,
        agent=result.agent_name,
        response=result.response,
        confidence=result.confidence,
        escalated=result.escalate,
        tool_calls=result.tool_calls_made if req.include_trace else [],
        elapsed_ms=result.elapsed_ms,
        facility_id=req.facility_id,
    )


@app.post("/agents/{agent_name}", response_model=QueryResponse)
async def query_agent(
    agent_name: str,
    req: AgentQueryRequest,
    api_key: str = Depends(get_api_key),
):
    """Talk directly to a specific specialist agent, bypassing the coordinator."""
    agent = _agents.get(agent_name)
    if agent is None:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_name}' not found. Available: thermal, power, capacity, incident, coordinator",
        )

    try:
        result = agent.run(task=req.task, context=req.context)
    except Exception as e:
        logger.exception(f"Agent {agent_name} failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        query=req.task,
        agent=result.agent_name,
        response=result.response,
        confidence=result.confidence,
        escalated=result.escalate,
        tool_calls=result.tool_calls_made,
        elapsed_ms=result.elapsed_ms,
        facility_id="",
    )


@app.get("/incidents", response_model=IncidentResponse)
async def get_incidents(
    facility_id: str = "default",
    min_severity: str = "P3",
    api_key: str = Depends(get_api_key),
):
    """Get current open incidents."""
    incident_agent = _agents.get("incident")
    if not incident_agent:
        raise HTTPException(status_code=503, detail="Agents not initialised")

    result = incident_agent.execute_tool(
        "get_open_incidents",
        {"min_severity": min_severity},
    )
    return IncidentResponse(
        incidents=result.get("incidents", []),
        count=result.get("count", 0),
        facility_id=facility_id,
    )


@app.get("/memory/search", response_model=MemorySearchResponse)
async def search_memory(
    q: str,
    top_k: int = 5,
    api_key: str = Depends(get_api_key),
):
    """Semantic search over episodic memory (past incidents + resolutions)."""
    episodic = _agents.get("episodic_memory")
    if not episodic:
        return MemorySearchResponse(query=q, results=[], count=0)

    result = episodic.search_as_dict(q, top_k=top_k)
    return MemorySearchResponse(
        query=q,
        results=result.get("results", []),
        count=result.get("count", 0),
    )


@app.get("/agents/status")
async def agents_status(api_key: str = Depends(get_api_key)):
    """Heartbeat status of all agents in the monitoring loop."""
    working_mem = _agents.get("working_memory")
    if not working_mem:
        return {"agents": {}}
    return {"agents": working_mem.get_agent_status()}
