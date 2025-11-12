"""
working.py

Working memory using Redis.

Short-lived, fast. Used for:
- Agent heartbeats and status
- In-flight task state
- Recent agent findings (TTL 5 min)
- Active incidents (TTL 24h)
- Coordinator decisions

All values are JSON-serialised. Keys follow the convention:
  {agent_name}:{key}       - agent-owned state
  incident:{id}            - incident records
  coordinator:{decision}   - coordinator decisions
  facility:{id}:{metric}   - facility-level state
"""

import json
import logging
import time
from typing import Any, Optional
import redis

logger = logging.getLogger(__name__)

DEFAULT_TTL = 300  # 5 minutes


class WorkingMemory:
    """Redis-backed working memory shared across all agents."""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        try:
            self._redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self._redis.ping()
            logger.info(f"WorkingMemory connected to Redis {host}:{port}")
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Working memory disabled.")
            self._redis = None

    def write(self, key: str, value: Any, ttl: int = DEFAULT_TTL):
        """Write a value with optional TTL in seconds."""
        if not self._redis:
            return
        try:
            serialised = json.dumps(value)
            if ttl > 0:
                self._redis.setex(key, ttl, serialised)
            else:
                self._redis.set(key, serialised)
        except Exception as e:
            logger.debug(f"WorkingMemory write failed for {key}: {e}")

    def read(self, key: str) -> Optional[Any]:
        """Read a value. Returns None if not found or expired."""
        if not self._redis:
            return None
        try:
            raw = self._redis.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.debug(f"WorkingMemory read failed for {key}: {e}")
            return None

    def read_pattern(self, pattern: str) -> dict[str, Any]:
        """Read all keys matching a pattern. Returns {key: value} dict."""
        if not self._redis:
            return {}
        try:
            keys = self._redis.keys(pattern)
            if not keys:
                return {}
            values = self._redis.mget(keys)
            result = {}
            for key, raw in zip(keys, values):
                if raw is not None:
                    try:
                        result[key] = json.loads(raw)
                    except json.JSONDecodeError:
                        result[key] = raw
            return result
        except Exception as e:
            logger.debug(f"WorkingMemory pattern read failed for {pattern}: {e}")
            return {}

    def delete(self, key: str):
        if not self._redis:
            return
        try:
            self._redis.delete(key)
        except Exception:
            pass

    def increment(self, key: str, ttl: int = DEFAULT_TTL) -> int:
        """Atomic increment. Useful for counters."""
        if not self._redis:
            return 0
        try:
            val = self._redis.incr(key)
            self._redis.expire(key, ttl)
            return val
        except Exception:
            return 0

    def publish(self, channel: str, message: Any):
        """Publish a message to a Redis pub/sub channel."""
        if not self._redis:
            return
        try:
            self._redis.publish(channel, json.dumps(message))
        except Exception as e:
            logger.debug(f"Publish failed: {e}")

    def agent_heartbeat(self, agent_name: str):
        """Record that an agent is alive. Expires after 2 minutes."""
        self.write(f"heartbeat:{agent_name}", {"ts": time.time(), "agent": agent_name}, ttl=120)

    def get_agent_status(self) -> dict[str, dict]:
        """Get status of all agents based on heartbeats."""
        heartbeats = self.read_pattern("heartbeat:*")
        now = time.time()
        status = {}
        for key, val in heartbeats.items():
            agent_name = key.replace("heartbeat:", "")
            last_seen = val.get("ts", 0)
            age_s = now - last_seen
            status[agent_name] = {
                "alive": age_s < 90,
                "last_seen_s": round(age_s),
            }
        return status
HEARTBEAT_TTL_S = 120  # agent considered dead after 2 min without heartbeat
