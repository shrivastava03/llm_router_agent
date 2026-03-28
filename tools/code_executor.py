"""
tools/code_executor.py
─────────────────────────────────────────────────────────────────
Code Executor Tool

Safely executes Python code snippets in a subprocess with a hard
timeout. Uses subprocess isolation — not exec() — so runaway code
can be killed cleanly without taking down the main process.

No external dependencies beyond the standard library.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from typing import Optional

from config import TOOLS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    code:       str
    stdout:     str
    stderr:     str
    exit_code:  int
    timed_out:  bool = False
    error:      Optional[str] = None

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.error

    def as_text(self) -> str:
        if self.timed_out:
            return f"[Code execution timed out after {TOOLS.code_executor_timeout}s]"
        if self.error:
            return f"[Executor error: {self.error}]"
        out = []
        if self.stdout:
            out.append(f"stdout:\n{self.stdout.strip()}")
        if self.stderr:
            out.append(f"stderr:\n{self.stderr.strip()}")
        if not out:
            out.append("[No output]")
        status = "✓ exit 0" if self.exit_code == 0 else f"✗ exit {self.exit_code}"
        return f"[{status}]\n" + "\n".join(out)


# ─────────────────────────────────────────────────────────────────
# CodeExecutorTool
# ─────────────────────────────────────────────────────────────────

class CodeExecutorTool:
    """
    Agent tool: execute a Python code snippet and return its output.

    Runs code in a subprocess with:
      - hard timeout (default 10s from config)
      - stdout / stderr capture
      - non-zero exit code detection

    Usage:
        executor = CodeExecutorTool()
        result   = executor.run("print(sum(range(100)))")
        print(result.as_text())   # → stdout: 4950
    """

    def __init__(self, timeout: Optional[float] = None):
        self.timeout = timeout or TOOLS.code_executor_timeout

    def run(self, code: str) -> ExecutionResult:
        """Execute `code` and return stdout/stderr/exit_code."""
        code = textwrap.dedent(code).strip()
        if not code:
            return ExecutionResult(code=code, stdout="", stderr="",
                                   exit_code=0, error="empty code")

        # Write to a temp file — avoids shell injection
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(code)
                tmp_path = f.name

            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output = True,
                text           = True,
                timeout        = self.timeout,
            )
            logger.info(
                f"[CodeExecutor] exit={result.returncode} "
                f"stdout={len(result.stdout)}b stderr={len(result.stderr)}b"
            )
            return ExecutionResult(
                code      = code,
                stdout    = result.stdout,
                stderr    = result.stderr,
                exit_code = result.returncode,
            )

        except subprocess.TimeoutExpired:
            logger.warning(f"[CodeExecutor] Timed out after {self.timeout}s")
            return ExecutionResult(
                code=code, stdout="", stderr="",
                exit_code=-1, timed_out=True,
            )
        except Exception as e:
            logger.error(f"[CodeExecutor] Error: {e}")
            return ExecutionResult(
                code=code, stdout="", stderr="",
                exit_code=-1, error=str(e),
            )
