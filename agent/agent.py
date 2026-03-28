from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.classifier import ComplexityClassifier
from core.router import Router
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

    def summary(self) -> str:
        status = f"KILLED ({self.kill_reason})" if self.killed else "COMPLETED"
        lines = [
            f"\n{'═'*56}",
            f"  Task     : {self.task[:70]}",
            f"  Status   : {status}",
            f"  Steps    : {self.iterations}",
            f"  Tokens   : {self.total_tokens}",
            f"  Tier     : {self.tier_used}",
            f"{'─'*56}",
        ]
        for s in self.steps:
            lines.append(f"  Step {s.iteration}: [{s.tool}] {s.thought[:60]}")
        lines += [f"{'─'*56}", f"  Answer   : {self.answer[:200]}", f"{'═'*56}"]
        return "\n".join(lines)

class Agent:
    def __init__(
        self,
        classifier: Optional[ComplexityClassifier] = None,
        router: Optional[Router] = None,
        guard: Optional[BudgetGuard] = None,
        connector: Optional[HFConnector] = None,
        memory: Optional[AgentMemory] = None,
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
        logger.info("[Agent] Initialised.")

    async def setup(self) -> None:
        await self.guard.setup()

    async def run(
        self,
        task: str,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> AgentResult:
        await self.setup()

        clf = self.classifier.classify(task)
        decision = self.router.route(task, clf.final_score)
        
        past = self.memory.retrieve(task, top_k=config.MEMORY.max_results,
                                    score_threshold=config.MEMORY.score_threshold)
        mem_ctx = self.memory.build_context_string(past)

        builder = PromptBuilder()
        builder.set_memory_context(mem_ctx)
        builder.add_message("user", task)

        session = self.guard.open_session(session_id=session_id, task=task)
        result = AgentResult(task=task, tier_used=decision.tier.value)

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

        except BudgetExceededError as e:
            result.killed, result.kill_reason = True, "budget_exceeded"
            result.answer = "Task stopped - token budget exceeded."
        except LoopKillError as e:
            result.killed, result.kill_reason = True, e.reason
            result.answer = f"Task stopped - {e.reason}."
        finally:
            if session.is_alive:
                await self.guard.close_session(session)

        if result.answer and not result.killed:
            self.memory.store(task, result.answer, session_id or session.session_id, decision.tier.value, result.total_tokens)

        return result

async def _smoke_test():
    import os
    from dotenv import load_dotenv
    
    load_dotenv(dotenv_path=".env.example")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    test_logger = logging.getLogger(__name__)
    test_logger.info("Starting Agent Smoke Test...")

    if not os.getenv("GROQ_API_KEY"):
        print("\n[!] Error: GROQ_API_KEY not found in .env.example file.")
        return

    try:
        agent = Agent()
    except Exception as e:
        print(f"Failed to initialize Agent: {e}")
        return

    tasks = [
        "What is 17 * 26? Use the code executor.",
        "How many chances of Iran is in winning war? Is this a situation of WW3?"
    ]

    for task in tasks:
        print(f"\n{'='*60}\nRUNNING TASK: {task}\n{'='*60}")
        try:
            result = await agent.run(task)
            print(result.summary())
        except Exception as e:
            print(f"Error during task execution: {e}")

if __name__ == "__main__":
    import asyncio
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(_smoke_test())