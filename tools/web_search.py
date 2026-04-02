"""
tools/web_search.py
─────────────────────────────────────────────────────────────────
Web Search Tool using Tavily (Industry standard for AI Agents).
Requires: pip install tavily-python
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from tavily import TavilyClient
    _TAVILY_AVAILABLE = True
except ImportError:
    _TAVILY_AVAILABLE = False
    logger.warning("tavily-python not installed. Run: pip install tavily-python")


# ─────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────

@dataclass
class SearchResponse:
    query:   str
    results: str
    error:   Optional[str] = None

    def as_context(self) -> str:
        """Format all results for prompt injection."""
        if self.error:
            if "STOP SEARCHING" in self.error:
                return f"[CRITICAL ERROR: {self.error}]"
            return f"[Web search failed: {self.error}. Do not retry this exact search.]"
            
        if not self.results:
            return f"[No results found for: '{self.query}'.]"
        
        return f"### Web search results for '{self.query}':\n\n{self.results}"


# ─────────────────────────────────────────────────────────────────
# WebSearchTool
# ─────────────────────────────────────────────────────────────────

class WebSearchTool:
    """
    Agent tool: search the web via Tavily API.
    """

    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.api_key = os.getenv("TAVILY_API_KEY")
        self.client = TavilyClient(api_key=self.api_key) if self.api_key and _TAVILY_AVAILABLE else None

    def search(self, query: str) -> SearchResponse:
        if not _TAVILY_AVAILABLE:
            return SearchResponse(query=query, results="", error="tavily-python not installed")
        if not self.client:
            return SearchResponse(query=query, results="", error="TAVILY_API_KEY not found in .env")
        if not query.strip():
            return SearchResponse(query=query, results="", error="empty query")

        logger.info(f"[WebSearch] Searching Tavily for: '{query}'")
        
        try:
            # Ask Tavily to generate a concrete answer from the sites it scrapes
            response = self.client.search(
                query=query, 
                search_depth="advanced", 
                max_results=self.max_results,
                include_answer=True
            )
            
            results_text = ""
            
            # Put the Direct Answer right at the top so the agent sees it first
            direct_answer = response.get("answer")
            if direct_answer:
                results_text += f"💡 DIRECT ANSWER FROM SEARCH ENGINE:\n{direct_answer}\n\n"
                results_text += "--- Supporting Sources ---\n"

            for r in response.get("results", []):
                content = r.get("content", "")
                # Truncate to save LLM tokens and prevent context bloat
                if len(content) > 400:
                    content = content[:400] + "... [truncated]"
                results_text += f"[{r.get('title', 'No Title')}]\n{content}\nSource: {r.get('url', '')}\n\n"
            
            if not results_text.strip():
                return SearchResponse(query=query, results="", error="No results returned")

            logger.info(f"[WebSearch] Success for '{query}'")
            return SearchResponse(query=query, results=results_text)

        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"[WebSearch] Error: {e}")
            # Protect against runaway loops if Tavily API quota is exceeded
            if any(term in error_msg for term in ["429", "quota", "rate limit", "401", "403", "unauthorized"]):
                return SearchResponse(
                    query=query, 
                    results="", 
                    error="API Quota exceeded or Rate Limited. STOP SEARCHING immediately and use the 'finish' tool to inform the user."
                )
            return SearchResponse(query=query, results="", error=str(e))