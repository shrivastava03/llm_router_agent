"""
tools/web_search.py
─────────────────────────────────────────────────────────────────
Web Search Tool

Uses DuckDuckGo — completely free, no API key required.
Returns structured SearchResult objects the agent can reason over.

Install:
  pip install duckduckgo-search
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from duckduckgo_search import DDGS
    _DDG_AVAILABLE = True
except ImportError:
    _DDG_AVAILABLE = False
    logger.warning("duckduckgo-search not installed. Run: pip install duckduckgo-search")


# ─────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    title:   str
    url:     str
    snippet: str

    def as_text(self) -> str:
        return f"[{self.title}]\n{self.snippet}\nSource: {self.url}"


@dataclass
class SearchResponse:
    query:   str
    results: list[SearchResult]
    error:   Optional[str] = None

    def as_context(self) -> str:
        """Format all results for prompt injection."""
        if self.error:
            return f"[Web search failed: {self.error}]"
        if not self.results:
            return f"[No results found for: {self.query}]"
        blocks = "\n\n".join(r.as_text() for r in self.results)
        return f"### Web search results for '{self.query}':\n\n{blocks}"


# ─────────────────────────────────────────────────────────────────
# WebSearchTool
# ─────────────────────────────────────────────────────────────────

class WebSearchTool:
    """
    Agent tool: search the web via DuckDuckGo.

    Usage:
        tool   = WebSearchTool(max_results=5)
        result = tool.search("latest Python async best practices")
        print(result.as_context())   # inject into prompt
    """

    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def search(self, query: str) -> SearchResponse:
        if not _DDG_AVAILABLE:
            return SearchResponse(query=query, results=[], error="duckduckgo-search not installed")

        if not query.strip():
            return SearchResponse(query=query, results=[], error="empty query")

        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=self.max_results):
                    results.append(SearchResult(
                        title   = r.get("title",   ""),
                        url     = r.get("href",    ""),
                        snippet = r.get("body",    ""),
                    ))

            logger.info(f"[WebSearch] '{query}' → {len(results)} results")
            return SearchResponse(query=query, results=results)

        except Exception as e:
            logger.error(f"[WebSearch] Error: {e}")
            return SearchResponse(query=query, results=[], error=str(e))
