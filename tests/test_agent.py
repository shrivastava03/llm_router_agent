"""
tests/test_agent.py
─────────────────────────────────────────────────────────────────
End-to-end tests for the Agent ReAct loop.

Strategy: mock HFConnector and ToolDispatcher so tests run
instantly with no API calls, no network, no Groq key needed.
Every test validates agent behaviour, not LLM output.

Run:
    pytest tests/test_agent.py -v
"""

import sys
import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from core.budget_guard import BudgetConfig, BudgetGuard
from core.classifier   import ComplexityClassifier
from core.router       import Router, ModelTier
from core.memory       import AgentMemory
from hf_connector      import LLMResponse
from agent.agent       import Agent, AgentResult
from agent.tool_dispatcher import ParsedAction, DispatchResult


# ─────────────────────────────────────────────────────────────────
# Helpers — build mock objects
# ─────────────────────────────────────────────────────────────────

def make_llm_response(text: str, tier: ModelTier = ModelTier.SIMPLE, error: str = None) -> LLMResponse:
    """Construct a realistic LLMResponse without calling any API."""
    return LLMResponse(
        text           = text,
        model_id       = "llama-3.1-8b-instant",
        tier_requested = tier,
        tier_used      = tier,
        tokens_input   = 100,
        tokens_output  = 50,
        latency_ms     = 200.0,
        error          = error,
    )


def make_agent(
    llm_replies:    list[str],
    dispatch_results: list[DispatchResult],
    budget_cfg:     Optional[BudgetConfig] = None,
) -> Agent:
    """
    Build an Agent with mocked components for instant, reliable testing.
    """
    mock_connector = MagicMock()
    mock_connector._responses = [make_llm_response(r) for r in llm_replies]

    # ✅ FIX 1: Simulate the connector actually talking to the guard!
    # By manually triggering guard.checkpoint(), tokens and iterations are tracked properly.
    async def fake_complete_with_guard(*args, **kwargs):
        session = kwargs.get('session')
        guard = kwargs.get('guard')
        response = mock_connector._responses.pop(0)
        
        # Trigger the guard limits so it can kill the agent if needed
        await guard.checkpoint(session, tokens_used=response.total_tokens, output=response.text)
        return response

    mock_connector.complete_with_guard = fake_complete_with_guard

    # Mock dispatcher — parse() returns valid actions, dispatch() returns observations
    mock_dispatcher        = MagicMock()
    mock_dispatcher.parse  = MagicMock(side_effect=[
        ParsedAction(thought=f"step {i}", tool=r.tool, input=r.observation, raw="")
        for i, r in enumerate(dispatch_results, 1)
    ])
    mock_dispatcher.dispatch = MagicMock(side_effect=dispatch_results)

    guard = BudgetGuard(budget_cfg or BudgetConfig(
        max_tokens_per_session = 50_000,
        max_iterations         = 10,
        repetition_threshold   = 0.99,
        db_path                = "test_audit.db", 
    ))

    # ✅ FIX 2: Stub out the async SQLite auditing to prevent "Event loop is closed" crashes
    guard.setup = AsyncMock()
    guard._log = MagicMock()
    guard._log.log_event = AsyncMock()
    guard._log.upsert_session = AsyncMock()

    # Disabled memory
    mock_memory          = MagicMock(spec=AgentMemory)
    mock_memory.retrieve = MagicMock(return_value=[])
    mock_memory.build_context_string = MagicMock(return_value="")
    mock_memory.store    = MagicMock()
    mock_memory.count    = MagicMock(return_value=0)

    agent             = Agent(
        classifier = ComplexityClassifier(),
        router     = Router(),
        guard      = guard,
        connector  = mock_connector,
        memory     = mock_memory,
    )
    agent.dispatcher  = mock_dispatcher
    return agent


def run(coro):
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────
# 1. Happy path — single tool call then finish
# ─────────────────────────────────────────────────────────────────

def test_single_tool_then_finish():
    """Agent calls one tool then returns a final answer cleanly."""
    dispatch_results = [
        DispatchResult(tool="code_executor", observation="391", is_finish=False),
        DispatchResult(tool="finish",        observation="17 * 23 = 391", is_finish=True),
    ]
    agent  = make_agent(
        llm_replies      = ['{"thought":"compute","tool":"code_executor","tool_input":"print(17*23)"}',
                            '{"thought":"done","tool":"finish","tool_input":"17 * 23 = 391"}'],
        dispatch_results = dispatch_results,
    )
    result = run(agent.run("What is 17 * 23? Use the code executor."))

    assert not result.killed
    assert result.answer == "17 * 23 = 391"
    assert result.iterations == 2
    assert len(result.steps) == 2
    assert result.steps[0].tool == "code_executor"
    assert result.steps[1].tool == "finish"


# ─────────────────────────────────────────────────────────────────
# 2. Multi-step — web search → finish
# ─────────────────────────────────────────────────────────────────

def test_web_search_then_finish():
    """Agent searches the web then synthesises a final answer."""
    dispatch_results = [
        DispatchResult(tool="web_search", observation="Python 3.13 released Oct 2024", is_finish=False),
        DispatchResult(tool="finish",     observation="Latest Python is 3.13", is_finish=True),
    ]
    agent  = make_agent(
        llm_replies      = ['{"thought":"search","tool":"web_search","tool_input":"latest Python version"}',
                            '{"thought":"answer","tool":"finish","tool_input":"Latest Python is 3.13"}'],
        dispatch_results = dispatch_results,
    )
    result = run(agent.run("What is the latest Python version?"))

    assert not result.killed
    assert result.answer == "Latest Python is 3.13"
    assert result.steps[0].tool == "web_search"
    assert result.total_tokens == 300   # 2 steps × (100 in + 50 out)


# ─────────────────────────────────────────────────────────────────
# 3. Budget — token cap kills the session
# ─────────────────────────────────────────────────────────────────

def test_budget_token_cap_kills_agent():
    """Agent is killed when cumulative tokens exceed the session cap."""
    # Each LLM call costs 150 tokens (100 in + 50 out). Cap at 200 → kills on 2nd call.
    tight_budget = BudgetConfig(
        max_tokens_per_session = 200,
        max_iterations         = 20,
        repetition_threshold   = 0.99,
        db_path                = "test_audit.db",
    )
    dispatch_results = [
        DispatchResult(tool="web_search", observation="result 1", is_finish=False),
        DispatchResult(tool="web_search", observation="result 2", is_finish=False),
        DispatchResult(tool="finish",     observation="done",     is_finish=True),
    ]
    agent  = make_agent(
        llm_replies      = ['{"thought":"s","tool":"web_search","tool_input":"q"}'] * 3,
        dispatch_results = dispatch_results,
        budget_cfg       = tight_budget,
    )
    result = run(agent.run("Some complex research task"))

    assert result.killed
    assert result.kill_reason == "budget_exceeded"


# ─────────────────────────────────────────────────────────────────
# 4. Budget — iteration cap kills the session
# ─────────────────────────────────────────────────────────────────

def test_iteration_cap_kills_agent():
    """Agent is killed after hitting max_iterations without finishing."""
    tight_budget = BudgetConfig(
        max_tokens_per_session = 100_000,
        max_iterations         = 3,       # very tight
        repetition_threshold   = 0.99,
        db_path                = "test_audit.db",
    )
    # Never returns finish — keeps searching
    dispatch_results = [
        DispatchResult(tool="web_search", observation=f"result {i}", is_finish=False)
        for i in range(10)
    ]
    agent  = make_agent(
        llm_replies      = ['{"thought":"s","tool":"web_search","tool_input":"q"}'] * 10,
        dispatch_results = dispatch_results,
        budget_cfg       = tight_budget,
    )
    result = run(agent.run("Keep searching forever"))

    assert result.killed
    assert "iteration" in result.kill_reason


# ─────────────────────────────────────────────────────────────────
# 5. LLM error — connector returns error field
# ─────────────────────────────────────────────────────────────────

def test_llm_error_surfaces_gracefully():
    """If the LLM connector returns an error, agent surfaces it cleanly."""
    mock_connector = MagicMock()
    mock_connector.complete_with_guard = AsyncMock(
        return_value = make_llm_response("", error="503 Service Unavailable")
    )
    guard = BudgetGuard(BudgetConfig(db_path="test_audit.db"))
    
    # Stub logging to prevent SQLite errors during manual agent initialization
    guard.setup = AsyncMock()
    guard._log = MagicMock()
    guard._log.upsert_session = AsyncMock()
    guard._log.log_event = AsyncMock()  # ✅ Added the missing mock here

    mock_memory = MagicMock(spec=AgentMemory)
    mock_memory.retrieve = MagicMock(return_value=[])
    mock_memory.build_context_string = MagicMock(return_value="")
    mock_memory.store = MagicMock()
    mock_memory.count = MagicMock(return_value=0)

    agent = Agent(
        classifier = ComplexityClassifier(),
        router     = Router(),
        guard      = guard,
        connector  = mock_connector,
        memory     = mock_memory,
    )

    result = run(agent.run("Do something"))

    assert "Model error" in result.answer
    assert "503" in result.answer
    assert not result.killed   # error ≠ budget kill — different code path


# ─────────────────────────────────────────────────────────────────
# 6. Routing — simple prompt goes to simple tier
# ─────────────────────────────────────────────────────────────────

def test_simple_prompt_routes_to_simple_tier():
    """A clearly simple prompt is classified and routed to the cheap model."""
    dispatch_results = [
        DispatchResult(tool="finish", observation="Bonjour", is_finish=True),
    ]
    agent  = make_agent(
        llm_replies      = ['{"thought":"translate","tool":"finish","tool_input":"Bonjour"}'],
        dispatch_results = dispatch_results,
    )
    result = run(agent.run("Translate 'Hello' to French."))

    assert result.tier_used == ModelTier.SIMPLE.value
    assert not result.killed


# ─────────────────────────────────────────────────────────────────
# 7. Routing — complex prompt goes to complex tier
# ─────────────────────────────────────────────────────────────────

def test_complex_prompt_routes_to_complex_tier():
    """A deeply technical prompt is classified and routed to the heavy model."""
    dispatch_results = [
        DispatchResult(tool="finish", observation="Analysis complete", is_finish=True),
    ]
    agent  = make_agent(
        llm_replies      = ['{"thought":"analyze","tool":"finish","tool_input":"Analysis complete"}'],
        dispatch_results = dispatch_results,
    )
    result = run(agent.run(
        "Analyze the time and space complexity of this merge sort variant "
        "and suggest cache-friendly optimizations with benchmarks."
    ))

    assert result.tier_used == ModelTier.COMPLEX.value
    assert not result.killed


# ─────────────────────────────────────────────────────────────────
# 8. AgentResult.summary() — output format is correct
# ─────────────────────────────────────────────────────────────────

def test_agent_result_summary_format():
    """AgentResult.summary() contains all expected fields."""
    dispatch_results = [
        DispatchResult(tool="finish", observation="42", is_finish=True),
    ]
    agent  = make_agent(
        llm_replies      = ['{"thought":"done","tool":"finish","tool_input":"42"}'],
        dispatch_results = dispatch_results,
    )
    result  = run(agent.run("What is 6 * 7?"))
    summary = result.summary()

    assert "COMPLETED" in summary
    assert "42" in summary
    assert "Steps" in summary
    assert "Tokens" in summary


# ─────────────────────────────────────────────────────────────────
# 9. Memory store — called after successful completion
# ─────────────────────────────────────────────────────────────────

def test_memory_store_called_on_success():
    """Completed interactions are persisted to memory."""
    dispatch_results = [
        DispatchResult(tool="finish", observation="The answer is 42", is_finish=True),
    ]
    agent  = make_agent(
        llm_replies      = ['{"thought":"done","tool":"finish","tool_input":"The answer is 42"}'],
        dispatch_results = dispatch_results,
    )
    run(agent.run("What is the answer?", session_id="test-session"))

    agent.memory.store.assert_called_once()
    call_kwargs = agent.memory.store.call_args
    assert "The answer is 42" in str(call_kwargs)


# ─────────────────────────────────────────────────────────────────
# 10. Memory store — NOT called when session is killed
# ─────────────────────────────────────────────────────────────────

def test_memory_not_stored_on_kill():
    """Killed sessions should not be persisted to memory."""
    tight_budget = BudgetConfig(
        max_tokens_per_session = 50,   # kills immediately
        max_iterations         = 20,
        repetition_threshold   = 0.99,
        db_path                = "test_audit.db",
    )
    dispatch_results = [
        DispatchResult(tool="web_search", observation="result", is_finish=False),
    ] * 5
    agent  = make_agent(
        llm_replies      = ['{"thought":"s","tool":"web_search","tool_input":"q"}'] * 5,
        dispatch_results = dispatch_results,
        budget_cfg       = tight_budget,
    )
    result = run(agent.run("Some task"))

    assert result.killed
    agent.memory.store.assert_not_called()