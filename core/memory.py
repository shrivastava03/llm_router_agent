"""
core/memory.py
─────────────────────────────────────────────────────────────────
Memory Module — ChromaDB Wrapper

Gives the agent persistent short-term + long-term memory.
Every completed session is stored as a vector. On each new
request, the top-K most similar past interactions are retrieved
and injected into the prompt as context.
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


class AgentMemory:
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
        """Standard fire-and-forget database write."""
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

    async def store_with_transaction(
        self,
        prompt: str,
        response: str,
        session_id: str,
        tier_used: str = "unknown",
        tokens: int = 0,
    ) -> Optional[str]:
        """
        Store with transaction semantics: all-or-nothing.
        Verifies that the entry was actually written before returning.
        """
        if not self._ready:
            logger.warning("[Memory] ChromaDB not ready — cannot store")
            return None

        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            prompt=prompt,
            response=response,
            session_id=session_id,
            tier_used=tier_used,
            tokens=tokens,
            timestamp=time.time(),
        )

        try:
            # Step 1: Write to ChromaDB
            logger.debug(f"[Memory] Writing {entry.id[:12]}…")
            self._col.add(
                ids=[entry.id],
                documents=[entry.to_document()],
                metadatas=[entry.to_metadata()],
            )

            # Step 2: Verify write succeeded (critical!)
            logger.debug(f"[Memory] Verifying {entry.id[:12]}…")
            verification = self._col.get(ids=[entry.id])
            
            if not verification["documents"] or len(verification["documents"]) == 0:
                raise RuntimeError(
                    f"Verification failed: entry {entry.id[:12]}… not found after write"
                )
            
            # Step 3: Validate retrieved entry matches what we wrote
            stored_doc = verification["documents"][0]
            expected_doc = entry.to_document()
            
            if stored_doc != expected_doc:
                raise RuntimeError(
                    f"Verification failed: stored document doesn't match\n"
                    f"  Expected: {expected_doc[:100]}\n"
                    f"  Stored: {stored_doc[:100]}"
                )
            
            logger.info(f"[Memory] ✓ Transactional store VERIFIED: {entry.id[:12]}…")
            return entry.id

        except Exception as e:
            logger.error(
                f"[Memory] ✗ Transactional store FAILED for {entry.id[:12]}…: {e}"
            )
            return None

    # ── Retrieve ───────────────────────────────────────────────────

    def retrieve(
        self,
        query:           str,
        top_k:           int   = 5,
        score_threshold: float = 0.75,
    ) -> list[MemoryResult]:
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
        if not results:
            return ""
        blocks = [r.as_context_block() for r in results]
        header = "### Relevant past interactions (for context):\n"
        return header + "\n\n".join(blocks) + "\n\n---\n\n"

    # ── Stats ──────────────────────────────────────────────────────

    def count(self) -> int:
        if not hasattr(self, '_col') or self._col is None:
            return 0
        return self._col.count()

    def clear(self) -> None:
        if not self._ready:
            return
        ids = self._col.get()["ids"]
        if ids:
            self._col.delete(ids=ids)
            logger.warning(f"[Memory] Cleared {len(ids)} entries.")