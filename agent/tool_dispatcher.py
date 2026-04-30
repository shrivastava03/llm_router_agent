from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

from tools.web_search import WebSearchTool
from tools.code_executor import CodeExecutorTool
from tools.file_reader import FileReaderTool
from core.memory import AgentMemory

logger = logging.getLogger(__name__)

@dataclass
class ParsedAction:
    thought: str
    tool: str
    input: str
    raw: str
    parse_error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.parse_error is None and bool(self.tool)

    @property
    def is_finish(self) -> bool:
        return self.tool == "finish"

@dataclass
class DispatchResult:
    tool: str
    observation: str
    is_finish: bool = False
    error: Optional[str] = None

class ToolDispatcher:
    
    # ─────────────────────────────────────────────────────────
    # Timeout limits per tool (seconds)
    # ─────────────────────────────────────────────────────────
    TIMEOUT_SECONDS = {
        "code_executor": 10,
        "web_search": 15,
        "file_reader": 5,
        "memory_search": 5,
    }
    
    # ─────────────────────────────────────────────────────────
    # Rate limiting (minimum ms between calls)
    # ─────────────────────────────────────────────────────────
    RATE_LIMIT_MS = {
        "web_search": 100,      # Max 1 search per 100ms
        "code_executor": 50,    # Max 1 execution per 50ms
        "file_reader": 50,      # Max 1 read per 50ms
    }

    def __init__(
        self,
        web_search: Optional[WebSearchTool] = None,
        code_executor: Optional[CodeExecutorTool] = None,
        file_reader: Optional[FileReaderTool] = None,
        memory: Optional[AgentMemory] = None,
    ):
        self._web = web_search or WebSearchTool()
        self._code = code_executor or CodeExecutorTool()
        self._files = file_reader or FileReaderTool()
        self._memory = memory
        self._last_invocation = {}

    def parse(self, llm_output: str) -> ParsedAction:
        raw = llm_output.strip()

        # Clean unwanted injected text
        clean_text = re.sub(
            r"http://googleusercontent\.com/immersive_entry_chip/0",
            "",
            raw,
        )

        # Extract JSON block if present
        brace = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if brace:
            clean_text = brace.group(0)

        try:
            data = json.loads(clean_text)
            return ParsedAction(
                thought=str(data.get("thought", "")),
                tool=str(data.get("tool", "")).strip().lower(),
                input=str(data.get("tool_input", data.get("input", ""))),
                raw=llm_output,
            )
        except json.JSONDecodeError as e:
            logger.warning(
                f"[Dispatcher] JSON parse failed: {e}\nRaw: {llm_output[:200]}"
            )
            return ParsedAction(
                thought="", tool="", input="", raw=llm_output, parse_error=str(e),
            )

    def dispatch(self, action: ParsedAction) -> DispatchResult:
        if not action.is_valid:
            return DispatchResult(
                tool="unknown",
                observation=(
                    "[Parse error] Could not understand the action. "
                    f"Reason: {action.parse_error}. "
                    "Please respond with valid JSON using the required format."
                ),
                error=action.parse_error,
            )

        logger.info(f"[Dispatcher] tool={action.tool} input={action.input[:80]}")

        if action.is_finish:
            return DispatchResult(
                tool="finish",
                observation=action.input,
                is_finish=True,
            )

        if action.tool == "web_search":
            return self._run_web_search(action.input)
        if action.tool == "code_executor":
            return self._run_code(action.input)
        if action.tool == "file_reader":
            return self._run_file_reader(action.input)
        if action.tool == "memory_search":
            return self._run_memory_search(action.input)

        available = ["web_search", "code_executor", "file_reader", "memory_search", "finish"]
        return DispatchResult(
            tool=action.tool,
            observation=f"[Unknown tool: '{action.tool}'] Available tools: {', '.join(available)}",
            error=f"unknown tool: {action.tool}",
        )

    # ── Hardened Execution Methods ─────────────────────────────────

    def _apply_rate_limit(self, tool_name: str) -> None:
        """Enforce minimum time between invocations."""
        if tool_name not in self.RATE_LIMIT_MS:
            return
        
        min_interval_ms = self.RATE_LIMIT_MS[tool_name]
        last_time = self._last_invocation.get(tool_name, 0)
        elapsed_ms = (time.time() - last_time) * 1000
        
        if elapsed_ms < min_interval_ms:
            sleep_ms = min_interval_ms - elapsed_ms
            logger.debug(f"[ToolDispatcher] Rate limit: sleeping {sleep_ms:.0f}ms")
            time.sleep(sleep_ms / 1000)
        
        self._last_invocation[tool_name] = time.time()

    def _run_code(self, code: str) -> DispatchResult:
        self._apply_rate_limit("code_executor")
        logger.info(f"[ToolDispatcher] Executing code (timeout: 10s, sandbox: /tmp)")
        
        try:
            result = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                timeout=self.TIMEOUT_SECONDS["code_executor"],
                cwd="/tmp",          # Isolated directory
                env={},              # Blank environment (No API keys exposed)
                text=True,           # Return strings
            )
            
            if result.returncode == 0:
                return DispatchResult(
                    tool="code_executor", observation=result.stdout.strip(), error=None
                )
            else:
                return DispatchResult(
                    tool="code_executor",
                    observation=f"[Code execution failed with status {result.returncode}]",
                    error=result.stderr.strip(),
                )
        except subprocess.TimeoutExpired:
            logger.error("[ToolDispatcher] Code execution TIMEOUT (>10s)")
            return DispatchResult(
                tool="code_executor",
                observation="[Code execution timeout after 10 seconds]",
                error="timeout",
            )
        except Exception as e:
            logger.error(f"[ToolDispatcher] Code execution failed: {e}")
            return DispatchResult(tool="code_executor", observation="[Code execution failed]", error=str(e))

    def _run_web_search(self, query: str) -> DispatchResult:
        self._apply_rate_limit("web_search")
        if not query.strip():
            return DispatchResult(tool="web_search", observation="[Empty search query]", error="empty_query")
        
        logger.info(f"[ToolDispatcher] Web search: '{query[:100]}…'")
        try:
            result = self._web.search(query)
            if result is None:
                raise ValueError("Search returned None")
            
            context = result.as_context()
            if not isinstance(context, str) or not context.strip():
                return DispatchResult(tool="web_search", observation="[Search returned no results]", error=None)
            
            return DispatchResult(tool="web_search", observation=context, error=result.error)
        except Exception as e:
            logger.error(f"[ToolDispatcher] Web search failed: {e}")
            return DispatchResult(tool="web_search", observation="[Web search failed]", error=str(e))

    def _run_file_reader(self, path: str) -> DispatchResult:
        self._apply_rate_limit("file_reader")
        logger.info(f"[ToolDispatcher] Reading file: {path}")
        try:
            result = self._files.read(path)
            return DispatchResult(tool="file_reader", observation=result.as_context(), error=result.error)
        except Exception as e:
            logger.error(f"[ToolDispatcher] File read failed: {e}")
            return DispatchResult(tool="file_reader", observation="[File read failed]", error=str(e))

    def _run_memory_search(self, query: str) -> DispatchResult:
        if not self._memory:
            return DispatchResult(tool="memory_search", observation="[Memory not available in this session]")

        results = self._memory.retrieve(query)
        ctx = self._memory.build_context_string(results)
        obs = ctx if ctx else "[No relevant memories found]"

        return DispatchResult(tool="memory_search", observation=obs)