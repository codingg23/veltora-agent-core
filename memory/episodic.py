"""
episodic.py

Episodic memory using pgvector for semantic search over past incidents.

Each memory entry is:
- A text description of what happened
- A vector embedding of that text (for similarity search)
- Metadata: resolution, severity, affected scope, timestamp

When an agent encounters a new situation, it searches episodic memory for
similar past events. If there's a match with a known resolution, it can
surface that path immediately rather than reasoning from scratch.

This is what makes the system get better over time. Each resolved incident
is written back, so the agents learn from real ops experience.

Requires: PostgreSQL with pgvector extension, and the anthropic client
for generating embeddings.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional
import anthropic

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "voyage-3"  # via Anthropic API
EMBEDDING_DIM = 1024


@dataclass
class MemoryEntry:
    id: str
    text: str
    metadata: dict
    similarity: float = 0.0


class EpisodicMemory:
    """
    Semantic memory layer backed by pgvector.

    Stores incidents, resolutions, and learnings as vector embeddings.
    Supports semantic search: "thermal anomaly in GPU row" will surface
    similar past incidents even if the exact words don't match.
    """

    def __init__(self, connection_string: str):
        self.conn_str = connection_string
        self._client = anthropic.Anthropic()
        self._conn = None
        self._setup()

    def _setup(self):
        try:
            import psycopg2
            from pgvector.psycopg2 import register_vector
            self._conn = psycopg2.connect(self.conn_str)
            register_vector(self._conn)
            self._ensure_schema()
            logger.info("EpisodicMemory connected to pgvector")
        except ImportError:
            logger.warning("psycopg2/pgvector not installed. Episodic memory disabled.")
        except Exception as e:
            logger.error(f"EpisodicMemory setup failed: {e}")

    def _ensure_schema(self):
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding vector(%d),
                    metadata JSONB,
                    created_at DOUBLE PRECISION
                )
            """ % EMBEDDING_DIM)
            cur.execute("CREATE INDEX IF NOT EXISTS episodes_vec_idx ON episodes USING ivfflat (embedding vector_cosine_ops)")
            self._conn.commit()

    def _embed(self, text: str) -> list[float]:
        """Generate embedding for text using Anthropic's embedding model."""
        try:
            response = self._client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text,
            )
            return response.embeddings[0].embedding
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return [0.0] * EMBEDDING_DIM

    def write(self, text: str, metadata: dict) -> str:
        """Write a new episode to memory. Returns the episode ID."""
        if not self._conn:
            return ""

        episode_id = f"ep-{int(time.time())}-{hash(text) % 10000:04d}"
        embedding = self._embed(text)

        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO episodes (id, text, embedding, metadata, created_at) VALUES (%s, %s, %s, %s, %s)"
                    " ON CONFLICT (id) DO UPDATE SET text=EXCLUDED.text, embedding=EXCLUDED.embedding, metadata=EXCLUDED.metadata",
                    (episode_id, text, embedding, json.dumps(metadata), time.time()),
                )
                self._conn.commit()
            logger.debug(f"Wrote episode {episode_id}")
            return episode_id
        except Exception as e:
            logger.error(f"Episode write failed: {e}")
            self._conn.rollback()
            return ""

    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.6) -> list[MemoryEntry]:
        """
        Semantic search over episodic memory.

        Returns the most similar past episodes to the query.
        """
        if not self._conn:
            return []

        query_embedding = self._embed(query)

        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, text, metadata,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM episodes
                    WHERE 1 - (embedding <=> %s::vector) > %s
                    ORDER BY similarity DESC
                    LIMIT %s
                    """,
                    (query_embedding, query_embedding, min_similarity, top_k),
                )
                rows = cur.fetchall()

            results = []
            for row_id, text, metadata_json, sim in rows:
                meta = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
                results.append(MemoryEntry(
                    id=row_id,
                    text=text,
                    metadata=meta or {},
                    similarity=float(sim),
                ))
            return results

        except Exception as e:
            logger.error(f"Episode search failed: {e}")
            return []

    def search_as_dict(self, query: str, top_k: int = 5) -> dict:
        """Return search results as a dict (for tool call responses)."""
        results = self.search(query, top_k=top_k)
        return {
            "query": query,
            "results": [
                {
                    "id": e.id,
                    "text": e.text,
                    "similarity": round(e.similarity, 3),
                    "metadata": e.metadata,
                }
                for e in results
            ],
            "count": len(results),
        }
