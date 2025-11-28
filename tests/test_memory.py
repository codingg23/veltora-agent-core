"""
test_memory.py

Tests for working and episodic memory layers.

Working memory tests use a real Redis instance if available,
otherwise skip gracefully. Episodic memory tests mock the DB.
"""

import json
import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from memory.working import WorkingMemory


class TestWorkingMemory:
    """Tests for Redis-backed working memory."""

    @pytest.fixture
    def mem(self):
        """Return a WorkingMemory instance. Uses real Redis if available."""
        return WorkingMemory(host="localhost", port=6379)

    def test_write_and_read_string(self, mem):
        if mem._redis is None:
            pytest.skip("Redis not available")
        key = f"test:str:{int(time.time())}"
        mem.write(key, "hello world", ttl=10)
        assert mem.read(key) == "hello world"

    def test_write_and_read_dict(self, mem):
        if mem._redis is None:
            pytest.skip("Redis not available")
        key = f"test:dict:{int(time.time())}"
        data = {"agent": "thermal", "temp_c": 26.5, "ts": time.time()}
        mem.write(key, data, ttl=10)
        result = mem.read(key)
        assert result["agent"] == "thermal"
        assert abs(result["temp_c"] - 26.5) < 0.001

    def test_read_nonexistent_key_returns_none(self, mem):
        if mem._redis is None:
            pytest.skip("Redis not available")
        assert mem.read("definitely:does:not:exist:xyz123") is None

    def test_ttl_expiry(self, mem):
        if mem._redis is None:
            pytest.skip("Redis not available")
        key = f"test:ttl:{int(time.time())}"
        mem.write(key, "expires soon", ttl=1)
        time.sleep(2)
        assert mem.read(key) is None

    def test_read_pattern(self, mem):
        if mem._redis is None:
            pytest.skip("Redis not available")
        prefix = f"test:pattern:{int(time.time())}"
        mem.write(f"{prefix}:a", {"v": 1}, ttl=10)
        mem.write(f"{prefix}:b", {"v": 2}, ttl=10)
        mem.write(f"{prefix}:c", {"v": 3}, ttl=10)

        results = mem.read_pattern(f"{prefix}:*")
        assert len(results) == 3
        values = {v["v"] for v in results.values()}
        assert values == {1, 2, 3}

    def test_delete(self, mem):
        if mem._redis is None:
            pytest.skip("Redis not available")
        key = f"test:del:{int(time.time())}"
        mem.write(key, "to be deleted", ttl=60)
        assert mem.read(key) == "to be deleted"
        mem.delete(key)
        assert mem.read(key) is None

    def test_agent_heartbeat(self, mem):
        if mem._redis is None:
            pytest.skip("Redis not available")
        mem.agent_heartbeat("ThermalAgent")
        status = mem.get_agent_status()
        assert "ThermalAgent" in status
        assert status["ThermalAgent"]["alive"] is True
        assert status["ThermalAgent"]["last_seen_s"] < 5

    def test_stale_heartbeat_shows_not_alive(self, mem):
        if mem._redis is None:
            pytest.skip("Redis not available")
        # Write a heartbeat with old timestamp
        key = "heartbeat:StaleAgent"
        mem._redis.setex(key, 30, json.dumps({"ts": time.time() - 120, "agent": "StaleAgent"}))
        status = mem.get_agent_status()
        assert status.get("StaleAgent", {}).get("alive") is False

    def test_increment(self, mem):
        if mem._redis is None:
            pytest.skip("Redis not available")
        key = f"test:counter:{int(time.time())}"
        v1 = mem.increment(key, ttl=10)
        v2 = mem.increment(key, ttl=10)
        v3 = mem.increment(key, ttl=10)
        assert v1 == 1
        assert v2 == 2
        assert v3 == 3

    def test_no_redis_write_is_noop(self):
        """WorkingMemory with no Redis should not raise - silently does nothing."""
        mem = WorkingMemory.__new__(WorkingMemory)
        mem._redis = None
        mem.write("key", "value")  # should not raise
        assert mem.read("key") is None
        assert mem.read_pattern("*") == {}


class TestWorkingMemoryMocked:
    """Tests that work without a real Redis connection using mocks."""

    def _make_mem(self):
        mem = WorkingMemory.__new__(WorkingMemory)
        mem._redis = MagicMock()
        return mem

    def test_write_serialises_to_json(self):
        mem = self._make_mem()
        mem.write("mykey", {"x": 1}, ttl=60)
        mem._redis.setex.assert_called_once_with("mykey", 60, json.dumps({"x": 1}))

    def test_write_no_ttl_uses_set(self):
        mem = self._make_mem()
        mem.write("mykey", "val", ttl=0)
        mem._redis.set.assert_called_once_with("mykey", json.dumps("val"))

    def test_read_deserialises_json(self):
        mem = self._make_mem()
        mem._redis.get.return_value = json.dumps({"result": 42})
        result = mem.read("mykey")
        assert result == {"result": 42}

    def test_read_returns_none_on_miss(self):
        mem = self._make_mem()
        mem._redis.get.return_value = None
        assert mem.read("missing") is None

    def test_read_pattern_returns_dict(self):
        mem = self._make_mem()
        mem._redis.keys.return_value = ["a:1", "a:2"]
        mem._redis.mget.return_value = [json.dumps({"v": 1}), json.dumps({"v": 2})]
        result = mem.read_pattern("a:*")
        assert len(result) == 2
        assert result["a:1"] == {"v": 1}
        assert result["a:2"] == {"v": 2}

    def test_redis_error_returns_none_not_exception(self):
        mem = self._make_mem()
        mem._redis.get.side_effect = Exception("connection refused")
        result = mem.read("key")  # should not raise
        assert result is None
