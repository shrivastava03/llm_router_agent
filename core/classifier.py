"""
core/classifier.py
─────────────────────────────────────────────────────────────────
Complexity Classifier — the brain of the LLM Router Agent.

Scores an incoming prompt on a 0.0–1.0 scale using 4 weighted
signals, then maps the score to a model tier:
  score < 0.40  →  SIMPLE  (fast, cheap model)
  score ≥ 0.40  →  COMPLEX (heavy, capable model)

All models run locally / via HuggingFace free Inference API.
No OpenAI key required.
"""

from __future__ import annotations
from config import ModelTier
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from config      import CLASSIFIER, ClassifierSettings   # ← pull from config.py

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Backward-compat alias so any code that imports ClassifierConfig
# from this module still works
# ─────────────────────────────────────────────────────────────────
ClassifierConfig = ClassifierSettings


# ─────────────────────────────────────────────────────────────────
# Anchor Prompts for Embedding Signal
# ─────────────────────────────────────────────────────────────────

SIMPLE_ANCHOR_PROMPTS = [
    "Translate this sentence to French.",
    "Fix the grammar in this paragraph.",
    "Summarize this text in one sentence.",
    "Convert this date to ISO format.",
    "What is the capital of Japan?",
    "Rewrite this email to be more formal.",
    "Extract all email addresses from this text.",
    "List the items in this text as bullet points.",
    "Correct the spelling in this paragraph.",
    "Format this JSON to be human-readable.",
]

COMPLEX_ANCHOR_PROMPTS = [
    "Analyze the time complexity of this algorithm and suggest optimizations.",
    "Debug this Python async code — it deadlocks under high concurrency.",
    "Compare these two system architectures and recommend one for a fintech startup.",
    "Explain why gradient descent gets stuck in saddle points and how to escape them.",
    "Design a database schema for a multi-tenant SaaS application.",
    "What are the security implications of this OAuth2 implementation?",
    "Write a Rust implementation of a lock-free queue with detailed comments.",
    "Audit this Solidity contract for reentrancy vulnerabilities.",
    "Given these five research papers, synthesize the key disagreements.",
    "Refactor this legacy codebase to follow SOLID principles.",
]


# ─────────────────────────────────────────────────────────────────
# Keyword Signal Lists
# ─────────────────────────────────────────────────────────────────

COMPLEX_KEYWORDS = {
    "analyze", "analyse", "compare", "contrast", "evaluate", "assess",
    "critique", "debate", "argue", "justify", "prove", "disprove",
    "explain why", "reason about", "trade-off", "trade off",
    "debug", "refactor", "optimize", "optimise", "architect", "design",
    "implement", "build", "audit", "review", "vulnerability", "security",
    "performance", "scalability", "concurrency", "async", "race condition",
    "step by step", "end to end", "comprehensive", "in depth", "detailed",
    "thoroughly", "complete guide", "walk me through",
    "synthesize", "synthesise", "research", "survey", "literature",
    "compare approaches", "pros and cons", "tradeoffs",
}

SIMPLE_KEYWORDS = {
    "translate", "fix grammar", "correct spelling", "format",
    "convert", "list", "bullet", "summarize", "summarise",
    "extract", "rewrite", "rephrase", "what is", "who is",
    "when did", "define", "spell check",
}


# ─────────────────────────────────────────────────────────────────
# Individual Signal Scorers
# ─────────────────────────────────────────────────────────────────

def _score_token_length(prompt: str, cfg: ClassifierSettings) -> float:
    estimated_tokens = len(prompt.split()) / 0.75
    if estimated_tokens <= cfg.token_len_simple_max:
        return 0.0
    if estimated_tokens >= cfg.token_len_complex_min:
        return 1.0
    span = cfg.token_len_complex_min - cfg.token_len_simple_max
    return (estimated_tokens - cfg.token_len_simple_max) / span


def _score_keywords(prompt: str) -> float:
    lower        = prompt.lower()
    complex_hits = sum(1 for kw in COMPLEX_KEYWORDS if kw in lower)
    simple_hits  = sum(1 for kw in SIMPLE_KEYWORDS  if kw in lower)
    raw          = (min(complex_hits, 4) / 4) - (min(simple_hits, 4) / 4)
    return max(0.0, min(1.0, (raw + 1.0) / 2))


def _score_structural_cues(prompt: str) -> float:
    score = 0.0
    if re.search(r"(\ba\)|\bb\)|\bc\)|1\.|2\.|3\.)", prompt):
        score += 0.3
    if "```" in prompt or re.search(r"`[^`]+`", prompt):
        score += 0.35
    q_count = prompt.count("?")
    if q_count >= 3:
        score += 0.25
    elif q_count == 2:
        score += 0.10
    bullet_lines = len(re.findall(r"^\s*[-*•]\s", prompt, re.MULTILINE))
    if bullet_lines >= 3:
        score += 0.2
    elif bullet_lines >= 1:
        score += 0.1
    return min(1.0, score)


# ─────────────────────────────────────────────────────────────────
# Embedding Signal
# ─────────────────────────────────────────────────────────────────

class EmbeddingSignal:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not _ST_AVAILABLE:
            self._ready = False
            return
        self._model = SentenceTransformer(model_name)
        self._simple_anchors = self._model.encode(SIMPLE_ANCHOR_PROMPTS, convert_to_numpy=True)
        self._complex_anchors = self._model.encode(COMPLEX_ANCHOR_PROMPTS, convert_to_numpy=True)
        self._ready = True

    def score(self, prompt: str) -> float:
        if not self._ready: return 0.5
        vec = self._model.encode([prompt], convert_to_numpy=True)[0]
        sim_simple  = float(np.mean(self._cosine_batch(vec, self._simple_anchors)))
        sim_complex = float(np.mean(self._cosine_batch(vec, self._complex_anchors)))
        total = sim_simple + sim_complex
        return sim_complex / total if total > 1e-9 else 0.5

    @staticmethod
    def _cosine_batch(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        vec_norm = vec / (np.linalg.norm(vec) + 1e-9)
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
        return matrix_norm @ vec_norm

# ─────────────────────────────────────────────────────────────────
# Result & Classifier
# ─────────────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    prompt:           str
    final_score:      float
    tier:             ModelTier
    signal_token_len: float
    signal_keyword:   float
    signal_embedding: float
    signal_structural:float
    reasoning:        str = ""

class ComplexityClassifier:
    def __init__(self, cfg: Optional[ClassifierSettings] = None):
        self.cfg = cfg or CLASSIFIER
        self._embedding = EmbeddingSignal(self.cfg.embedding_model_name)

    def classify(self, prompt: str) -> ClassificationResult:
        p = prompt.strip()
        if not p: raise ValueError("Prompt must not be empty.")

        s_len = _score_token_length(p, self.cfg)
        s_kw  = _score_keywords(p)
        s_emb = self._embedding.score(p)
        s_struct = _score_structural_cues(p)

        final_score = round(float(np.clip(
            self.cfg.weight_token_length * s_len
            + self.cfg.weight_keyword    * s_kw
            + self.cfg.weight_embedding  * s_emb
            + self.cfg.weight_structural * s_struct,
            0.0, 1.0,
        )), 4)

        tier = ModelTier.COMPLEX if final_score >= self.cfg.complexity_threshold else ModelTier.SIMPLE

        reasoning = f"Final Score: {final_score} | Threshold: {self.cfg.complexity_threshold}"

        return ClassificationResult(
            prompt=p, final_score=final_score, tier=tier,
            signal_token_len=s_len, signal_keyword=s_kw,
            signal_embedding=s_emb, signal_structural=s_struct,
            reasoning=reasoning
        )

# ─────────────────────────────────────────────────────────────────
# Smoke test — python core/classifier.py
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    logging.basicConfig(level=logging.INFO)

    test_prompts = [
        "Translate this paragraph to Spanish.",
        "Fix the grammar in this sentence.",
        "What is the capital of France?",
        "Convert 100 USD to EUR.",
        "Analyze the time and space complexity of this merge sort variant.",
        "Debug this async Python code — it intermittently deadlocks under concurrent writes. ```...```",
        "Compare REST vs GraphQL vs gRPC. Consider: a) latency b) developer experience c) schema evolution.",
        "Design a multi-tenant database schema for a SaaS billing system.",
    ]

    clf = ComplexityClassifier()
    print("\n" + "═"*52)
    print("  COMPLEXITY CLASSIFIER — SMOKE TEST")
    print("═"*52)
    for p in test_prompts:
        print(clf.classify(p))