from __future__ import annotations

import logging
import sys
import asyncio
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import warnings

warnings.filterwarnings("ignore", category=ResourceWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.classifier import ComplexityClassifier
from core.router import Router, ModelTier
from core.budget_guard import BudgetGuard, BudgetExceededError, LoopKillError
from core.memory import AgentMemory
from hf_connector import HFConnector
from agent.prompt_builder import PromptBuilder
from agent.tool_dispatcher import ToolDispatcher
import config

logger = logging.getLogger(__name__)

@dataclass
class AgentStep:
    iteration: int
    thought: str
    tool: str
    tool_input: str
    observation: str

@dataclass
class AgentResult:
    task: str
    answer: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    total_tokens: int = 0
    iterations: int = 0
    tier_used: str = ""
    killed: bool = False
    kill_reason: Optional[str] = None
    prompt_hash: Optional[str] = None

    def summary(self) -> str:
        status = f"KILLED ({self.kill_reason})" if self.killed else "COMPLETED"
        lines = [
            f"\n{'═'*56}",
            f"   Task     : {self.task[:70]}",
            f"   Status   : {status}",
            f"   Steps    : {self.iterations}",
            f"   Tokens   : {self.total_tokens}",
            f"   Tier     : {self.tier_used}",
            f"{'─'*56}",
        ]
        for s in self.steps:
            lines.append(f"   Step {s.iteration}: [{s.tool}] {s.thought[:60]}")
        lines += [f"{'─'*56}", f"   Answer   : {self.answer[:200]}", f"{'═'*56}"]
        return "\n".join(lines)

class Agent:
    def __init__(
        self,
        classifier: Optional[ComplexityClassifier] = None,
        router: Optional[Router] = None,
        guard: Optional[BudgetGuard] = None,
        connector: Optional[HFConnector] = None,
        memory: Optional[AgentMemory] = None,
        cache_ttl_minutes: int = 30,
    ):
        self.classifier = classifier or ComplexityClassifier()
        self.router = router or Router()
        self.guard = guard or BudgetGuard()
        self.connector = connector or HFConnector()
        self.memory = memory or AgentMemory(
            persist_dir=config.MEMORY.persist_dir,
            collection=config.MEMORY.collection,
        )
        self.dispatcher = ToolDispatcher(memory=self.memory)
        
        self._response_cache = {} 
        self._cache_ttl = timedelta(minutes=cache_ttl_minutes)
        logger.info("[Agent] Ready with Ensemble Classifier + Online Learning.")

    async def setup(self) -> None:
        await self.guard.setup()

    async def run(
        self,
        task: str,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        use_cache: bool = True,
    ) -> AgentResult:
        
        # 1. Check Response Cache
        task_hash = hashlib.sha256(task.encode()).hexdigest()
        if use_cache and task_hash in self._response_cache:
            cached_result, cached_time = self._response_cache[task_hash]
            if (datetime.now() - cached_time) < self._cache_ttl:
                logger.info(f"[Agent] ✓ Cache HIT for {task_hash[:12]}...")
                return cached_result

        await self.setup()

        # 2. Parallelize classification and memory retrieval correctly
        clf_task, past_task = await asyncio.gather(
            asyncio.to_thread(self.classifier.classify, task),
            asyncio.to_thread(
                self.memory.retrieve,
                task, 
                top_k=config.MEMORY.max_results,
                score_threshold=config.MEMORY.score_threshold
            )
        )
        
        # Pass the whole object, the router will handle extraction safely
        decision = self.router.route(task, clf_task)
        
        # 3. Adaptive Context Injection
        if decision.tier == ModelTier.SIMPLE:
            mem_ctx = ""
            logger.info(f"[Agent] Tier: SIMPLE — skipping context to save tokens.")
        else:
            mem_ctx = self.memory.build_context_string(past_task)

        builder = PromptBuilder()
        builder.set_memory_context(mem_ctx)
        builder.add_message("user", task)

        session = self.guard.open_session(session_id=session_id, task=task)
        result = AgentResult(task=task, tier_used=decision.tier.value, prompt_hash=task_hash)

        try:
            for iteration in range(1, config.BUDGET.max_iterations + 1):
                logger.info(f"[Agent] ── Iteration {iteration} ──────────────")
                prompt = builder.build()

                llm_response = await self.connector.complete_with_guard(
                    prompt=prompt,
                    tier=decision.tier,
                    session=session,
                    guard=self.guard,
                    system_prompt=system_prompt or PromptBuilder.SYSTEM_INSTRUCTION,
                )

                result.total_tokens += llm_response.total_tokens
                result.iterations = iteration

                if llm_response.error:
                    result.answer = f"Model error: {llm_response.error}"
                    break

                action = self.dispatcher.parse(llm_response.text)
                dispatch = self.dispatcher.dispatch(action)

                result.steps.append(AgentStep(
                    iteration=iteration,
                    thought=action.thought,
                    tool=action.tool,
                    tool_input=action.input,
                    observation=dispatch.observation[:300],
                ))

                builder.add_message("assistant", llm_response.text)
                if not dispatch.is_finish:
                    builder.add_observation(action.tool, dispatch.observation)

                if dispatch.is_finish:
                    result.answer = dispatch.observation
                    break

        except BudgetExceededError:
            result.killed, result.kill_reason = True, "budget_exceeded"
            result.answer = "Task stopped - token budget exceeded."
        except LoopKillError as e:
            result.killed, result.kill_reason = True, e.reason
            result.answer = f"Task stopped - {e.reason}."
        finally:
            if session.is_alive:
                await self.guard.close_session(session)

        # 4. Result persistence and caching
        if result.answer and not result.killed:
            # Safely fallback to standard store if transactional isn't ready
            if hasattr(self.memory, 'store_with_transaction'):
                await self.memory.store_with_transaction(task, result.answer, session_id or session.session_id, decision.tier.value, result.total_tokens)
            else:
                self.memory.store(task, result.answer, session_id or session.session_id, decision.tier.value, result.total_tokens)
            
            self._response_cache[task_hash] = (result, datetime.now())
            
            # Online learning: add feedback
            if hasattr(self.classifier, 'add_feedback'):
                self.classifier.add_feedback(
                    task_hash,
                    decision.tier.value,
                    result.iterations,
                    result.total_tokens
                )

        return result