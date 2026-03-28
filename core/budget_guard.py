"""
budget_guard.py
─────────────────────────────────────────────────────────────────
Budget Guard + Loop Kill Switch

Three independent tripwires — any one fires the kill switch:
  1. Token spend     — cumulative tokens in a session vs hard cap
  2. Iteration cap   — max LLM calls per task
  3. Repetition lock — detects stuck agent via output hashing

Every kill event is written to an async SQLite audit log so you
can replay exactly what happened and why.

Usage (inside your agent loop):
    guard = BudgetGuard()
    session = guard.open_session(session_id="user-123", task="...")

    for step in range(MAX_STEPS):
        await guard.checkpoint(session, tokens_used=320, output=llm_reply)
        # ^ raises BudgetExceededError or LoopKillError if a limit fires

    await guard.close_session(session)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Deque, Optional

import aiosqlite   # pip install aiosqlite

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────────────────────────

@dataclass
class BudgetConfig:
    # ── Token budget ──────────────────────────────────────────────
    max_tokens_per_session: int   = 8_000    # hard cap per session
    warn_tokens_at:         int   = 6_000    # emit warning before kill

    # ── Iteration cap ─────────────────────────────────────────────
    max_iterations:         int   = 12       # max LLM calls per task

    # ── Repetition detector ───────────────────────────────────────
    repetition_window:      int   = 4        # last N outputs to compare
    repetition_threshold:   float = 0.85     # similarity ratio → kill

    # ── Audit log ─────────────────────────────────────────────────
    db_path:                str   = "budget_audit.db"


# ─────────────────────────────────────────────────────────────────
# 2. Custom Exceptions
# ─────────────────────────────────────────────────────────────────

class BudgetExceededError(RuntimeError):
    """Raised when token spend exceeds the session budget."""
    def __init__(self, session_id: str, tokens_used: int, limit: int):
        self.session_id = session_id
        self.tokens_used = tokens_used
        self.limit = limit
        super().__init__(
            f"[BudgetGuard] Session '{session_id}' killed — "
            f"token spend {tokens_used} exceeded limit {limit}."
        )


class LoopKillError(RuntimeError):
    """Raised when the iteration cap or repetition detector fires."""
    def __init__(self, session_id: str, reason: str):
        self.session_id = session_id
        self.reason = reason
        super().__init__(f"[BudgetGuard] Session '{session_id}' killed — {reason}")


# ─────────────────────────────────────────────────────────────────
# 3. Kill Reason Enum (for audit log)
# ─────────────────────────────────────────────────────────────────

class KillReason(str, Enum):
    TOKEN_BUDGET   = "token_budget_exceeded"
    ITERATION_CAP  = "iteration_cap_reached"
    REPETITION     = "repetition_detected"
    MANUAL         = "manual_abort"


# ─────────────────────────────────────────────────────────────────
# 4. Session State
# ─────────────────────────────────────────────────────────────────

@dataclass
class AgentSession:
    session_id:    str
    task:          str
    started_at:    float             = field(default_factory=time.time)
    tokens_used:   int               = 0
    iterations:    int               = 0
    is_alive:      bool              = True
    kill_reason:   Optional[str]     = None
    # Rolling window of recent output hashes for repetition detection
    _output_hashes: Deque[str]       = field(default_factory=lambda: deque(maxlen=10))

    def record_output(self, text: str) -> None:
        """Hash the output and push into the rolling window."""
        h = hashlib.md5(text.strip().lower().encode()).hexdigest()
        self._output_hashes.append(h)

    def repetition_ratio(self, window: int) -> float:
        """
        Fraction of the last `window` outputs that are identical
        to the most recent one. 1.0 means the agent is just repeating
        the same thing over and over.
        """
        if len(self._output_hashes) < 2:
            return 0.0
        recent = list(self._output_hashes)[-window:]
        latest = recent[-1]
        return sum(1 for h in recent if h == latest) / len(recent)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.started_at


# ─────────────────────────────────────────────────────────────────
# 5. Audit Log (async SQLite)
# ─────────────────────────────────────────────────────────────────

class AuditLog:
    """
    Async SQLite audit log. Stores every session open, checkpoint,
    warning, and kill event. Zero external dependencies beyond aiosqlite.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = asyncio.Lock()

    async def setup(self) -> None:
        """Create tables if they don't exist."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id           TEXT PRIMARY KEY,
                    task         TEXT,
                    started_at   REAL,
                    ended_at     REAL,
                    tokens_used  INTEGER,
                    iterations   INTEGER,
                    status       TEXT,
                    kill_reason  TEXT
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id   TEXT,
                    event_type   TEXT,
                    detail       TEXT,
                    tokens_used  INTEGER,
                    iteration    INTEGER,
                    timestamp    REAL
                )
            """)
            await db.commit()

    async def log_event(
        self,
        session_id: str,
        event_type: str,
        detail: str,
        tokens_used: int = 0,
        iteration: int = 0,
    ) -> None:
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """INSERT INTO events
                       (session_id, event_type, detail, tokens_used, iteration, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (session_id, event_type, detail, tokens_used, iteration, time.time()),
                )
                await db.commit()

    async def upsert_session(self, session: AgentSession, status: str) -> None:
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """INSERT INTO sessions
                       (id, task, started_at, ended_at, tokens_used, iterations, status, kill_reason)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(id) DO UPDATE SET
                           ended_at    = excluded.ended_at,
                           tokens_used = excluded.tokens_used,
                           iterations  = excluded.iterations,
                           status      = excluded.status,
                           kill_reason = excluded.kill_reason
                    """,
                    (
                        session.session_id,
                        session.task,
                        session.started_at,
                        time.time(),
                        session.tokens_used,
                        session.iterations,
                        status,
                        session.kill_reason,
                    ),
                )
                await db.commit()

    async def get_session_history(self, session_id: str) -> list[dict]:
        """Replay all events for a session — useful for debugging kills."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM events WHERE session_id = ? ORDER BY timestamp",
                (session_id,),
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def get_spend_summary(self) -> list[dict]:
        """Aggregate token spend per session — plug this into a dashboard."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT id, task, tokens_used, iterations,
                          status, kill_reason,
                          ROUND(ended_at - started_at, 2) AS duration_secs
                   FROM sessions ORDER BY started_at DESC LIMIT 50"""
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]


# ─────────────────────────────────────────────────────────────────
# 6. BudgetGuard — main class
# ─────────────────────────────────────────────────────────────────

class BudgetGuard:
    """
    Wraps every agent loop iteration with three independent
    kill-switch checks. Async-native for FastAPI compatibility.

    Lifecycle:
        guard   = BudgetGuard()
        await   guard.setup()                          # create DB tables

        session = guard.open_session("user-abc", task="Summarize…")
        try:
            for step in agent_loop:
                llm_output = await call_llm(prompt)
                await guard.checkpoint(
                    session,
                    tokens_used = count_tokens(llm_output),
                    output      = llm_output,
                )
        except (BudgetExceededError, LoopKillError) as e:
            logger.warning(str(e))
        finally:
            await guard.close_session(session)
    """

    def __init__(self, config: Optional[BudgetConfig] = None):
        self.cfg   = config or BudgetConfig()
        self._log  = AuditLog(self.cfg.db_path)
        self._sessions: dict[str, AgentSession] = {}

    async def setup(self) -> None:
        """Call once at app startup to initialise the audit DB."""
        await self._log.setup()
        logger.info(f"BudgetGuard ready. Audit DB: {self.cfg.db_path}")

    # ── Session management ─────────────────────────────────────────

    def open_session(
        self,
        session_id: Optional[str] = None,
        task: str = "",
    ) -> AgentSession:
        """
        Open a new session. Returns an AgentSession you pass to
        every subsequent checkpoint() call.
        """
        sid = session_id or str(uuid.uuid4())
        session = AgentSession(session_id=sid, task=task)
        self._sessions[sid] = session
        logger.info(f"[BudgetGuard] Session opened: {sid[:16]}… task='{task[:50]}'")

        # Fire-and-forget: log the open event (non-blocking)
        asyncio.ensure_future(
            self._log.log_event(sid, "session_open", task)
        )
        asyncio.ensure_future(
            self._log.upsert_session(session, "open")
        )
        return session

    async def close_session(self, session: AgentSession) -> None:
        """Mark session as completed (not killed)."""
        session.is_alive = False
        await self._log.upsert_session(session, "completed")
        self._sessions.pop(session.session_id, None)
        logger.info(
            f"[BudgetGuard] Session closed: {session.session_id[:16]}… "
            f"tokens={session.tokens_used} iters={session.iterations} "
            f"elapsed={session.elapsed_seconds:.1f}s"
        )

    # ── Main checkpoint ────────────────────────────────────────────

    async def checkpoint(
        self,
        session: AgentSession,
        tokens_used: int,
        output: str = "",
    ) -> None:
        """
        Call this AFTER every LLM invocation inside your agent loop.

        Parameters
        ----------
        session     : The AgentSession returned by open_session()
        tokens_used : Token count for THIS step (not cumulative)
        output      : The raw LLM output string for this step

        Raises
        ------
        BudgetExceededError  if token budget is blown
        LoopKillError        if iteration cap or repetition detected
        """
        if not session.is_alive:
            raise LoopKillError(session.session_id, "session already killed")

        session.tokens_used += tokens_used
        session.iterations  += 1
        if output:
            session.record_output(output)

        await self._log.log_event(
            session.session_id,
            "checkpoint",
            f"iter={session.iterations} step_tokens={tokens_used}",
            tokens_used=session.tokens_used,
            iteration=session.iterations,
        )

        # ── Tripwire 1: Token budget warning ──────────────────────
        if session.tokens_used >= self.cfg.warn_tokens_at:
            remaining = self.cfg.max_tokens_per_session - session.tokens_used
            logger.warning(
                f"[BudgetGuard] ⚠ Session {session.session_id[:16]}… "
                f"approaching budget — {remaining} tokens remaining."
            )
            await self._log.log_event(
                session.session_id, "budget_warning",
                f"remaining={remaining}",
                tokens_used=session.tokens_used,
                iteration=session.iterations,
            )

        # ── Tripwire 1: Token budget kill ─────────────────────────
        if session.tokens_used >= self.cfg.max_tokens_per_session:
            await self._kill(session, KillReason.TOKEN_BUDGET)
            raise BudgetExceededError(
                session.session_id,
                session.tokens_used,
                self.cfg.max_tokens_per_session,
            )

        # ── Tripwire 2: Iteration cap ─────────────────────────────
        if session.iterations >= self.cfg.max_iterations:
            await self._kill(session, KillReason.ITERATION_CAP)
            raise LoopKillError(
                session.session_id,
                f"iteration cap reached ({self.cfg.max_iterations})",
            )

        # ── Tripwire 3: Repetition detector ───────────────────────
        if output and session.iterations >= self.cfg.repetition_window:
            ratio = session.repetition_ratio(self.cfg.repetition_window)
            if ratio >= self.cfg.repetition_threshold:
                await self._kill(session, KillReason.REPETITION)
                raise LoopKillError(
                    session.session_id,
                    f"repetition detected (ratio={ratio:.2f}, "
                    f"window={self.cfg.repetition_window})",
                )

    # ── Manual abort ───────────────────────────────────────────────

    async def abort(self, session: AgentSession, reason: str = "") -> None:
        """Manually kill a session (e.g. user cancels the request)."""
        await self._kill(session, KillReason.MANUAL, extra=reason)
        raise LoopKillError(session.session_id, f"manual abort — {reason}")

    # ── Internal ───────────────────────────────────────────────────

    async def _kill(
        self,
        session: AgentSession,
        reason: KillReason,
        extra: str = "",
    ) -> None:
        session.is_alive    = False
        session.kill_reason = reason.value
        detail = f"{reason.value} {extra}".strip()

        logger.error(
            f"[BudgetGuard] 🔴 KILL {session.session_id[:16]}… "
            f"reason={reason.value} tokens={session.tokens_used} "
            f"iters={session.iterations}"
        )

        await self._log.log_event(
            session.session_id, "kill", detail,
            tokens_used=session.tokens_used,
            iteration=session.iterations,
        )
        await self._log.upsert_session(session, "killed")
        self._sessions.pop(session.session_id, None)

    # ── Convenience: live spend report ────────────────────────────

    async def spend_report(self) -> list[dict]:
        """Return the last 50 sessions with cost + kill summary."""
        return await self._log.get_spend_summary()

    async def replay_session(self, session_id: str) -> list[dict]:
        """Return the full event log for a past session."""
        return await self._log.get_session_history(session_id)


# ─────────────────────────────────────────────────────────────────
# 7. Smoke test — run with: python budget_guard.py
# ─────────────────────────────────────────────────────────────────

async def _smoke_test():
    logging.basicConfig(level=logging.INFO)
    print("\n" + "═"*54)
    print("  BUDGET GUARD — SMOKE TEST")
    print("═"*54)

    guard = BudgetGuard(BudgetConfig(
        max_tokens_per_session=1000,
        warn_tokens_at=800,
        max_iterations=6,
        repetition_window=3,
        repetition_threshold=0.85,
        db_path=":memory:",   # in-memory DB for test
    ))
    await guard.setup()

    # ── Test 1: Token budget exceeded ─────────────────────────────
    print("\n[Test 1] Token budget exceeded")
    s = guard.open_session(task="Summarize a very long document")
    try:
        for _ in range(10):
            await guard.checkpoint(s, tokens_used=250, output="Some response.")
    except BudgetExceededError as e:
        print(f"  ✓ Caught: {e}")
    finally:
        if s.is_alive:
            await guard.close_session(s)

    # ── Test 2: Iteration cap ──────────────────────────────────────
    print("\n[Test 2] Iteration cap")
    s = guard.open_session(task="Solve complex reasoning task")
    try:
        for _ in range(20):
            await guard.checkpoint(s, tokens_used=10, output=f"Step {_} done.")
    except LoopKillError as e:
        print(f"  ✓ Caught: {e}")
    finally:
        if s.is_alive:
            await guard.close_session(s)

    # ── Test 3: Repetition detector ───────────────────────────────
    print("\n[Test 3] Repetition detector (stuck agent)")
    s = guard.open_session(task="Write a blog post")
    try:
        outputs = [
            "Here is my draft...",
            "Here is my draft...",
            "Here is my draft...",
            "Here is my draft...",
        ]
        for out in outputs:
            await guard.checkpoint(s, tokens_used=50, output=out)
    except LoopKillError as e:
        print(f"  ✓ Caught: {e}")
    finally:
        if s.is_alive:
            await guard.close_session(s)

    # ── Test 4: Happy path (no kills) ─────────────────────────────
    print("\n[Test 4] Happy path — no kills")
    s = guard.open_session(task="Format this JSON")
    try:
        for i in range(3):
            await guard.checkpoint(s, tokens_used=100, output=f"Output step {i}")
        await guard.close_session(s)
        print("  ✓ Session completed cleanly")
    except (BudgetExceededError, LoopKillError) as e:
        print(f"  ✗ Unexpected kill: {e}")

    # ── Spend report ──────────────────────────────────────────────
    print("\n[Spend Report]")
    report = await guard.spend_report()
    for row in report:
        print(
            f"  {row['id'][:16]}…  task='{row['task'][:30]}'  "
            f"tokens={row['tokens_used']}  iters={row['iterations']}  "
            f"status={row['status']}  kill={row['kill_reason']}"
        )

    print("\n" + "═"*54)


if __name__ == "__main__":
    asyncio.run(_smoke_test())
