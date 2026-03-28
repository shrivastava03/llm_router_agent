"""
tools/file_reader.py
─────────────────────────────────────────────────────────────────
File Reader Tool

Reads PDFs and CSVs uploaded by the user and returns their
content as structured text the agent can reason over.

Install:
  pip install pymupdf pandas
  (pymupdf installs as `fitz`)
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from config import TOOLS

logger = logging.getLogger(__name__)

try:
    import fitz   # PyMuPDF
    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False
    logger.warning("PyMuPDF not installed. PDF reading disabled. Run: pip install pymupdf")

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False
    logger.warning("pandas not installed. CSV reading disabled. Run: pip install pandas")


# ─────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────

@dataclass
class FileReadResult:
    filename:   str
    file_type:  str           # "pdf" | "csv" | "txt" | "unknown"
    content:    str           # extracted text
    row_count:  Optional[int] = None   # for CSVs
    page_count: Optional[int] = None   # for PDFs
    error:      Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    def as_context(self) -> str:
        if self.error:
            return f"[File read error ({self.filename}): {self.error}]"
        meta = []
        if self.page_count:
            meta.append(f"{self.page_count} pages")
        if self.row_count:
            meta.append(f"{self.row_count} rows")
        header = f"### File: {self.filename} ({', '.join(meta)}):\n" if meta else f"### File: {self.filename}:\n"
        return header + self.content


# ─────────────────────────────────────────────────────────────────
# FileReaderTool
# ─────────────────────────────────────────────────────────────────

class FileReaderTool:
    """
    Agent tool: read PDFs, CSVs, and plain text files.

    Accepts either a file path (str / Path) or raw bytes + filename.
    Enforces a max file size from config.

    Usage:
        reader = FileReaderTool()

        # From path:
        result = reader.read("report.pdf")

        # From bytes (e.g. FastAPI UploadFile):
        result = reader.read_bytes(file_bytes, filename="data.csv")

        print(result.as_context())   # inject into prompt
    """

    def __init__(self, max_mb: Optional[float] = None):
        self.max_bytes = int((max_mb or TOOLS.file_reader_max_mb) * 1024 * 1024)

    def read(self, path: Union[str, Path]) -> FileReadResult:
        """Read a file from disk."""
        path = Path(path)
        if not path.exists():
            return FileReadResult(filename=path.name, file_type="unknown",
                                  content="", error="file not found")

        size = path.stat().st_size
        if size > self.max_bytes:
            return FileReadResult(
                filename=path.name, file_type="unknown", content="",
                error=f"file too large ({size / 1e6:.1f} MB > {self.max_bytes / 1e6:.0f} MB limit)",
            )

        data = path.read_bytes()
        return self.read_bytes(data, filename=path.name)

    def read_bytes(self, data: bytes, filename: str) -> FileReadResult:
        """Read from raw bytes. Used with FastAPI UploadFile."""
        if len(data) > self.max_bytes:
            return FileReadResult(
                filename=filename, file_type="unknown", content="",
                error=f"file too large ({len(data) / 1e6:.1f} MB limit is {self.max_bytes / 1e6:.0f} MB)",
            )

        ext = Path(filename).suffix.lower()

        if ext == ".pdf":
            return self._read_pdf(data, filename)
        elif ext == ".csv":
            return self._read_csv(data, filename)
        elif ext in (".txt", ".md", ".log", ".json", ".yaml", ".yml"):
            return self._read_text(data, filename)
        else:
            # Try plain text as a fallback
            return self._read_text(data, filename, file_type="unknown")

    # ── PDF ───────────────────────────────────────────────────────

    def _read_pdf(self, data: bytes, filename: str) -> FileReadResult:
        if not _PDF_AVAILABLE:
            return FileReadResult(filename=filename, file_type="pdf",
                                  content="", error="PyMuPDF not installed")
        try:
            doc   = fitz.open(stream=data, filetype="pdf")
            pages = [page.get_text() for page in doc]
            text  = "\n\n".join(pages).strip()
            logger.info(f"[FileReader] PDF '{filename}' — {len(pages)} pages, {len(text)} chars")
            return FileReadResult(
                filename   = filename,
                file_type  = "pdf",
                content    = text,
                page_count = len(pages),
            )
        except Exception as e:
            return FileReadResult(filename=filename, file_type="pdf",
                                  content="", error=str(e))

    # ── CSV ───────────────────────────────────────────────────────

    def _read_csv(self, data: bytes, filename: str) -> FileReadResult:
        if not _PANDAS_AVAILABLE:
            return FileReadResult(filename=filename, file_type="csv",
                                  content="", error="pandas not installed")
        try:
            df    = pd.read_csv(io.BytesIO(data))
            # Return a compact markdown table (first 50 rows to stay within token budget)
            preview = df.head(50).to_markdown(index=False)
            summary = (
                f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
                f"Columns: {', '.join(df.columns.tolist())}\n\n"
                f"{preview}"
            )
            logger.info(f"[FileReader] CSV '{filename}' — {df.shape[0]} rows × {df.shape[1]} cols")
            return FileReadResult(
                filename  = filename,
                file_type = "csv",
                content   = summary,
                row_count = df.shape[0],
            )
        except Exception as e:
            return FileReadResult(filename=filename, file_type="csv",
                                  content="", error=str(e))

    # ── Plain text ────────────────────────────────────────────────

    def _read_text(self, data: bytes, filename: str,
                   file_type: str = "txt") -> FileReadResult:
        try:
            text = data.decode("utf-8", errors="replace").strip()
            logger.info(f"[FileReader] Text '{filename}' — {len(text)} chars")
            return FileReadResult(filename=filename, file_type=file_type, content=text)
        except Exception as e:
            return FileReadResult(filename=filename, file_type=file_type,
                                  content="", error=str(e))
