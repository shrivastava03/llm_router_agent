"""
tests/test_budget_guard.py
─────────────────────────────────────────────────────────────────
Unit tests for the BudgetGuard.

Run:
  pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pytest_asyncio
from core.budget_guard import (
    BudgetGuard, BudgetConfig,
    BudgetExceededError, LoopKillError,
)

# 1. We inject 'tmp_path' into the fixture. Pytest automatically provides this!
@pytest_asyncio.fixture
async def guard(tmp_path):
    # Create a safe, temporary file path for this specific test run
    db_file = tmp_path / "test_budget.db"
    
    # 2. Define the config dynamically inside the fixture so it uses the safe path
    test_cfg = BudgetConfig(
        max_tokens_per_session = 500,
        warn_tokens_at         = 400,
        max_iterations         = 5,
        repetition_window      = 3,
        repetition_threshold   = 0.85,
        db_path                = str(db_file), 
    )
    
    g = BudgetGuard(test_cfg)
    await g.setup() # Tables are created on the real, temporary disk file
    return g


# ── Token budget ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_token_budget_exceeded(guard):
    session = guard.open_session(task="test")
    with pytest.raises(BudgetExceededError):
        for i in range(10):
            # Make the output unique so we don't trigger the repetition guard
            await guard.checkpoint(session, tokens_used=100, output=f"unique response {i}")


# ── Iteration cap ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_iteration_cap(guard):
    session = guard.open_session(task="test")
    with pytest.raises(LoopKillError):
        for i in range(20):
            await guard.checkpoint(session, tokens_used=5, output=f"step {i}")


# ── Repetition detector ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_repetition_detected(guard):
    session = guard.open_session(task="test")
    with pytest.raises(LoopKillError):
        for _ in range(5):
            await guard.checkpoint(session, tokens_used=5, output="same output every time")


# ── Happy path ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_happy_path(guard):
    session = guard.open_session(task="test")
    await guard.checkpoint(session, tokens_used=10, output="step 1")
    await guard.checkpoint(session, tokens_used=10, output="step 2")
    await guard.checkpoint(session, tokens_used=10, output="step 3")
    assert session.is_alive
    await guard.close_session(session)
    assert not session.is_alive


# ── Already-killed session ────────────────────────────────────────

@pytest.mark.asyncio
async def test_dead_session_raises(guard):
    session = guard.open_session(task="test")
    await guard.close_session(session)
    with pytest.raises(LoopKillError):
        await guard.checkpoint(session, tokens_used=1, output="x")