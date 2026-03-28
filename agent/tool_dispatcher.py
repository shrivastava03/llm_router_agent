from __future__ import annotations

import json
import logging
import re
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

    def parse(self, llm_output: str) -> ParsedAction:
        raw = llm_output.strip()

        # Clean unwanted injected text (FIXED)
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
                thought="",
                tool="",
                input="",
                raw=llm_output,
                parse_error=str(e),
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

    def _run_web_search(self, query: str) -> DispatchResult:
        result = self._web.search(query)
        return DispatchResult(
            tool="web_search",
            observation=result.as_context(),
            error=result.error,
        )

    def _run_code(self, code: str) -> DispatchResult:
        result = self._code.run(code)
        return DispatchResult(
            tool="code_executor",
            observation=result.as_text(),
            error=result.error,
        )

    def _run_file_reader(self, path: str) -> DispatchResult:
        result = self._files.read(path)
        return DispatchResult(
            tool="file_reader",
            observation=result.as_context(),
            error=result.error,
        )

    def _run_memory_search(self, query: str) -> DispatchResult:
        if not self._memory:
            return DispatchResult(
                tool="memory_search",
                observation="[Memory not available in this session]",
            )

        results = self._memory.retrieve(query)
        ctx = self._memory.build_context_string(results)
        obs = ctx if ctx else "[No relevant memories found]"

        return DispatchResult(
            tool="memory_search",
            observation=obs,
        )