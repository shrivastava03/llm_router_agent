from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Any

from config import SIMPLE_MODEL, COMPLEX_MODEL, ModelConfig, CONNECTOR, ModelTier

logger = logging.getLogger(__name__)

@dataclass
class RoutingDecision:
    tier: ModelTier
    model_cfg: ModelConfig
    complexity_score: float
    confidence: float = 0.8
    override_reason: Optional[str] = None

    def __str__(self) -> str:
        override = f" [override: {self.override_reason}]" if self.override_reason else ""
        return (
            f"RoutingDecision | tier={self.tier.value} | "
            f"score={self.complexity_score:.3f} | conf={self.confidence:.2f}{override}"
        )

def _apply_overrides(prompt: str, score: float) -> Optional[tuple[ModelTier, str]]:
    lower = prompt.lower()
    if "```" in prompt and any(kw in lower for kw in ("fix", "debug", "refactor", "optimize")):
        return ModelTier.COMPLEX, "code block + engineering keyword"
    pure_format_keywords = {"translate", "fix grammar", "correct spelling", "reformat", "convert to json", "convert to csv"}
    if any(kw in lower for kw in pure_format_keywords) and score < 0.55:
        return ModelTier.SIMPLE, "pure formatting task override"
    return None

class Router:
    def __init__(self, enable_overrides: bool = True):
        self._overrides_on = enable_overrides
        self.default_threshold = 0.40

    def route(self, task: str, classification_result: Any) -> RoutingDecision:
        override_reason: Optional[str] = None
        
        # 1. Safely extract values from classification_result
        if hasattr(classification_result, 'final_score'):
            score = classification_result.final_score
            tier = getattr(classification_result, 'tier', 
                           ModelTier.COMPLEX if score >= self.default_threshold else ModelTier.SIMPLE)
            confidence = getattr(classification_result, 'confidence', score)
        elif isinstance(classification_result, (float, int)):
            score = float(classification_result)
            tier = ModelTier.COMPLEX if score >= self.default_threshold else ModelTier.SIMPLE
            confidence = score
        else:
            logger.warning("[Router] Unrecognized classification result format. Defaulting to 0.5")
            score = 0.5
            tier = ModelTier.COMPLEX
            confidence = 0.5

        # 2. Apply Overrides
        if self._overrides_on:
            override = _apply_overrides(task, score)
            if override:
                tier, override_reason = override
                logger.info(f"[Router] Override: {override_reason} → {tier.value}")

        # 3. Map to specific Model configurations
        model_cfg = SIMPLE_MODEL if tier == ModelTier.SIMPLE else COMPLEX_MODEL

        decision = RoutingDecision(
            tier=tier,
            model_cfg=model_cfg,
            complexity_score=score,
            confidence=confidence,
            override_reason=override_reason,
        )
        logger.info(f"[Router] {decision}")
        return decision