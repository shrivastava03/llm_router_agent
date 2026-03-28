"""
core/memory.py
─────────────────────────────────────────────────────────────────
Memory Module — ChromaDB Wrapper

Gives the agent persistent short-term + long-term memory.
Every completed session is stored as a vector. On each new
request, the top-K most similar past interactions are retrieved
and injected into the prompt as context.

Install:
  pip install chromadb

ChromaDB runs fully local — no external service needed.
Data persists to .chromadb/ directory between restarts.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False
    logger.warning(
        "chromadb not installed — memory module disabled. "
        "Run: pip install chromadb"
    )


# ─────────────────────────────────────────────────────────────────
# MemoryEntry — what gets stored
# ─────────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    id:          str
    prompt:      str
    response:    str
    session_id:  str
    tier_used:   str
    tokens:      int
    timestamp:   float

    def to_document(self) -> str:
        """Text stored in ChromaDB — both prompt and response for richer retrieval."""
        return f"Q: {self.prompt}\nA: {self.response}"

    def to_metadata(self) -> dict:
        return {
            "session_id": self.session_id,
            "tier_used":  self.tier_used,
            "tokens":     self.tokens,
            "timestamp":  self.timestamp,
        }


# ─────────────────────────────────────────────────────────────────
# MemoryResult — what gets retrieved
# ─────────────────────────────────────────────────────────────────

@dataclass
class MemoryResult:
    prompt:     str
    response:   str
    similarity: float   # 0.0 = identical, 2.0 = completely different (L2 distance)
    session_id: str
    tier_used:  str

    def as_context_block(self) -> str:
        """Format for injection into a new prompt as prior context."""
        return f"[Past interaction — similarity {self.similarity:.2f}]\nQ: {self.prompt}\nA: {self.response}"


# ─────────────────────────────────────────────────────────────────
# AgentMemory — main class
# ─────────────────────────────────────────────────────────────────

class AgentMemory:
    """
    Persistent vector memory backed by ChromaDB.

    Usage:
        memory = AgentMemory()

        # After a completed LLM call:
        memory.store(prompt, response, session_id, tier_used, tokens)

        # Before a new LLM call — inject past context:
        past = memory.retrieve(new_prompt, top_k=3)
        context = memory.build_context_string(past)
        full_prompt = context + "\\n\\n" + new_prompt
    """

    def __init__(self, persist_dir: str = ".chromadb", collection: str = "agent_memory"):
        self._ready = False
        if not _CHROMA_AVAILABLE:
            return

        try:
            self._client = chromadb.PersistentClient(
                path     = persist_dir,
                settings = Settings(anonymized_telemetry=False),
            )
            self._col = self._client.get_or_create_collection(
                name     = collection,
                metadata = {"hnsw:space": "l2"},   # L2 distance for similarity
            )
            self._ready = True
            count = self._col.count()
            logger.info(f"[Memory] ChromaDB ready — {count} entries in '{collection}'")
        except Exception as e:
            logger.error(f"[Memory] ChromaDB init failed: {e}")

    # ── Store ──────────────────────────────────────────────────────

    def store(
        self,
        prompt:     str,
        response:   str,
        session_id: str,
        tier_used:  str  = "unknown",
        tokens:     int  = 0,
    ) -> Optional[str]:
        """
        Persist a completed interaction to memory.
        Returns the entry ID, or None if memory is unavailable.
        """
        if not self._ready:
            return None

        entry = MemoryEntry(
            id         = str(uuid.uuid4()),
            prompt     = prompt,
            response   = response,
            session_id = session_id,
            tier_used  = tier_used,
            tokens     = tokens,
            timestamp  = time.time(),
        )

        try:
            self._col.add(
                ids        = [entry.id],
                documents  = [entry.to_document()],
                metadatas  = [entry.to_metadata()],
            )
            logger.debug(f"[Memory] Stored entry {entry.id[:12]}…")
            return entry.id
        except Exception as e:
            logger.error(f"[Memory] Store failed: {e}")
            return None

    # ── Retrieve ───────────────────────────────────────────────────

    def retrieve(
        self,
        query:           str,
        top_k:           int   = 5,
        score_threshold: float = 0.75,
    ) -> list[MemoryResult]:
        """
        Find the most similar past interactions to `query`.

        Parameters
        ----------
        query           : The current user prompt
        top_k           : Max number of results to return
        score_threshold : Only return results with distance ≤ threshold
                          (lower = more similar in L2 space)

        Returns a list of MemoryResult sorted by relevance (closest first).
        """
        if not self._ready or self._col.count() == 0:
            return []

        try:
            results = self._col.query(
                query_texts = [query],
                n_results   = min(top_k, self._col.count()),
            )

            memory_results = []
            docs       = results["documents"][0]
            metadatas  = results["metadatas"][0]
            distances  = results["distances"][0]

            for doc, meta, dist in zip(docs, metadatas, distances):
                if dist > score_threshold:
                    continue   # too dissimilar — skip

                # Parse stored document back into prompt / response
                parts    = doc.split("\nA: ", 1)
                prompt_t = parts[0].replace("Q: ", "", 1) if parts else doc
                resp_t   = parts[1] if len(parts) > 1 else ""

                memory_results.append(MemoryResult(
                    prompt     = prompt_t,
                    response   = resp_t,
                    similarity = dist,
                    session_id = meta.get("session_id", ""),
                    tier_used  = meta.get("tier_used", ""),
                ))

            logger.debug(f"[Memory] Retrieved {len(memory_results)} results for query.")
            return memory_results

        except Exception as e:
            logger.error(f"[Memory] Retrieve failed: {e}")
            return []

    # ── Context builder ────────────────────────────────────────────

    @staticmethod
    def build_context_string(results: list[MemoryResult]) -> str:
        """
        Formats retrieved memories into a context block for prompt injection.
        Returns empty string if no results.
        """
        if not results:
            return ""
        blocks = [r.as_context_block() for r in results]
        header = "### Relevant past interactions (for context):\n"
        return header + "\n\n".join(blocks) + "\n\n---\n\n"

    # ── Stats ──────────────────────────────────────────────────────

    def count(self) -> int:
        """Total entries stored in memory."""
        if not self._ready:
            return 0
        return self._col.count()

    def clear(self) -> None:
        """Wipe all stored memories. Use with care."""
        if not self._ready:
            return
        ids = self._col.get()["ids"]
        if ids:
            self._col.delete(ids=ids)
            logger.warning(f"[Memory] Cleared {len(ids)} entries.")
