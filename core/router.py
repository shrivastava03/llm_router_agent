"""
core/router.py
─────────────────────────────────────────────────────────────────
Router — Model Selector Logic

Sits between the classifier and the HF connector.
Takes a ClassificationResult, applies any override rules,
and returns a resolved RoutingDecision with the exact model
config to use.

Separation of concerns:
  classifier.py  → what is the complexity score?
  router.py      → given that score, WHICH model do we call?
  hf_connector   → actually call it
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from config import SIMPLE_MODEL, COMPLEX_MODEL, ModelConfig, CONNECTOR, ModelTier

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# RoutingDecision — what the router hands downstream
# ─────────────────────────────────────────────────────────────────

@dataclass
class RoutingDecision:
    tier:           ModelTier
    model_cfg:      ModelConfig
    complexity_score: float
    override_reason: Optional[str] = None   # set when a rule overrides classifier

    def __str__(self) -> str:
        override = f" [override: {self.override_reason}]" if self.override_reason else ""
        return (
            f"RoutingDecision | tier={self.tier.value} "
            f"model={self.model_cfg.model_id} "
            f"score={self.complexity_score:.3f}{override}"
        )


# ─────────────────────────────────────────────────────────────────
# Override rules
# These fire BEFORE the classifier score is applied.
# Add your company-specific rules here.
# ─────────────────────────────────────────────────────────────────

def _apply_overrides(prompt: str, score: float) -> Optional[tuple[ModelTier, str]]:
    """
    Returns (ModelTier, reason) if an override rule fires, else None.

    Rules are evaluated in order — first match wins.
    """
    lower = prompt.lower()

    # Always route code-execution requests to the heavy model
    if "```" in prompt and any(kw in lower for kw in ("fix", "debug", "refactor", "optimize")):
        return ModelTier.COMPLEX, "code block + engineering keyword"

    # Force simple for pure format/translate tasks regardless of length
    pure_format_keywords = {"translate", "fix grammar", "correct spelling",
                             "reformat", "convert to json", "convert to csv"}
    if any(kw in lower for kw in pure_format_keywords) and score < 0.55:
        return ModelTier.SIMPLE, "pure formatting task override"

    return None


# ─────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────

class Router:
    """
    Resolves a complexity score + raw prompt into a RoutingDecision.

    Usage:
        router   = Router()
        decision = router.route(prompt, complexity_score)
        # decision.model_cfg → pass to HFConnector
        # decision.tier      → pass to BudgetGuard log
    """

    def __init__(self, enable_overrides: bool = True):
        self._overrides_on = enable_overrides

    def route(self, prompt: str, complexity_score: float) -> RoutingDecision:
        """
        Main routing method.

        1. Check override rules first
        2. Fall back to threshold-based tier from classifier score
        3. Resolve ModelConfig from config.py
        """
        override_reason: Optional[str] = None

        # ── Step 1: overrides ─────────────────────────────────────
        if self._overrides_on:
            override = _apply_overrides(prompt, complexity_score)
            if override:
                tier, override_reason = override
                logger.info(f"[Router] Override fired: {override_reason} → {tier.value}")
            else:
                tier = self._score_to_tier(complexity_score)
        else:
            tier = self._score_to_tier(complexity_score)

        # ── Step 2: resolve model config ──────────────────────────
        model_cfg = SIMPLE_MODEL if tier == ModelTier.SIMPLE else COMPLEX_MODEL

        decision = RoutingDecision(
            tier             = tier,
            model_cfg        = model_cfg,
            complexity_score = complexity_score,
            override_reason  = override_reason,
        )
        logger.info(f"[Router] {decision}")
        return decision

    @staticmethod
    def _score_to_tier(score: float) -> ModelTier:
        from config import CLASSIFIER
        return ModelTier.COMPLEX if score >= CLASSIFIER.complexity_threshold else ModelTier.SIMPLE
