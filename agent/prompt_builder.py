from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

@dataclass
class ToolDescriptor:
    name: str
    description: str
    input_hint: str 

    def as_text(self) -> str:
        return f"  - {self.name}: {self.description}\n    Input: {self.input_hint}"

TOOL_REGISTRY: list[ToolDescriptor] = [
    ToolDescriptor(
        name="web_search",
        description="Search the web for current information via DuckDuckGo/Tavily.",
        input_hint="a plain search query string",
    ),
    ToolDescriptor(
        name="code_executor",
        description="Execute a Python code snippet and return stdout/stderr.",
        input_hint="valid Python source code as a string",
    ),
    ToolDescriptor(
        name="file_reader",
        description="Read a PDF or CSV file and return its text content.",
        input_hint="absolute or relative file path as a string",
    ),
    ToolDescriptor(
        name="memory_search",
        description="Search past interactions stored in memory for relevant context.",
        input_hint="a natural language query string",
    ),
    ToolDescriptor(
        name="finish",
        description="Return the final answer to the user and end the agent loop.",
        input_hint="the complete final answer as a string",
    ),
]

@dataclass
class Message:
    role: str 
    content: str

    def as_text(self) -> str:
        label = {"user": "User", "assistant": "Assistant", "observation": "Observation"}
        return f"{label.get(self.role, self.role)}: {self.content}"

class PromptBuilder:
    SYSTEM_INSTRUCTION = """You are a precise AI assistant. You MUST respond with a single, valid JSON object for every turn. 
Do NOT wrap the JSON in markdown code blocks (e.g., do not use ```json). Just output the raw JSON.

You must use this exact JSON structure:
{
  "thought": "Your step-by-step reasoning about what to do next.",
  "tool": "The exact name of the tool to use (web_search, code_executor, file_reader, memory_search, finish)",
  "tool_input": "The exact string input to pass to the tool."
}

Example 1 (Using a tool):
{
  "thought": "I need to calculate 17 * 23.",
  "tool": "code_executor",
  "tool_input": "print(17 * 23)"
}

Example 2 (Finishing the task):
{
  "thought": "The code executor returned 391, which is the final answer.",
  "tool": "finish",
  "tool_input": "The answer is 391."
}

CRITICAL RULES:
1. Output ONLY valid JSON. No conversational text before or after the JSON.
2. If you know the answer or have completed the task, use the 'finish' tool IMMEDIATELY. Do not over-research.
3. NEVER write anything else before the JSON or after the JSON.
4. If a tool returns no results or fails twice in a row, DO NOT keep trying. Use the 'finish' tool to inform the user that the information cannot be found.
5. If you receive a '💡 DIRECT ANSWER FROM SEARCH ENGINE', evaluate if it answers the user's prompt. If it does, use the 'finish' tool on your very next turn.
6. [MEMORY CONTEXT] is ONLY for background knowledge. DO NOT blindly repeat past answers. You MUST fulfill any new specific constraints (like specific counts, numbers, or formats) requested in the current prompt.
"""

    def __init__(self):
        self._memory_ctx: str = ""
        self._history: list[Message] = []
        self._tools: list[ToolDescriptor] = TOOL_REGISTRY

    def set_memory_context(self, ctx: str) -> None:
        self._memory_ctx = ctx.strip()

    def add_message(self, role: str, content: str) -> None:
        self._history.append(Message(role=role, content=content))

    def add_observation(self, tool_name: str, result: str) -> None:
        obs = f"[{tool_name} result]\n{result.strip()}"
        self._history.append(Message(role="observation", content=obs))

    def clear_history(self) -> None:
        self._history.clear()
        self._memory_ctx = ""

    def build(self) -> str:
        parts: list[str] = []

        parts.append(f"[SYSTEM]\n{self.SYSTEM_INSTRUCTION}")

        tool_lines = "\n".join(t.as_text() for t in self._tools)
        parts.append(f"[AVAILABLE TOOLS]\n{tool_lines}")

        if self._memory_ctx:
            parts.append(f"[MEMORY CONTEXT]\n{self._memory_ctx}")

        if self._history:
            history_text = "\n".join(m.as_text() for m in self._history)
            parts.append(f"[CONVERSATION]\n{history_text}")

        parts.append("Assistant:")

        return "\n\n".join(parts)

    def history_length(self) -> int:
        return len(self._history)

    def last_user_message(self) -> Optional[str]:
        for m in reversed(self._history):
            if m.role == "user":
                return m.content
        return None