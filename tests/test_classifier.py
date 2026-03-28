"""
tests/test_classifier.py
─────────────────────────────────────────────────────────────────
Unit tests for the ComplexityClassifier.

Run:
  pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from core.classifier import ComplexityClassifier, ModelTier
from core.router import Router


@pytest.fixture(scope="module")
def classifier():
    return ComplexityClassifier()


@pytest.fixture(scope="module")
def router():
    return Router()


# ── Classifier: SIMPLE prompts ────────────────────────────────────

@pytest.mark.parametrize("prompt", [
    "Translate this to French.",
    "Fix the grammar in this sentence.",
    "What is the capital of France?",
    "Convert this date to ISO format.",
])
def test_simple_prompts(classifier, prompt):
    result = classifier.classify(prompt)
    assert result.tier == ModelTier.SIMPLE, (
        f"Expected SIMPLE for '{prompt}', got score={result.final_score:.3f}"
    )


# ── Classifier: COMPLEX prompts ───────────────────────────────────

@pytest.mark.parametrize("prompt", [
    "Analyze the time complexity of this merge sort variant and suggest optimizations.",
    "Debug this async Python code — it intermittently deadlocks under concurrent writes.",
    "Compare REST vs GraphQL vs gRPC for a high-throughput microservices system.",
])
def test_complex_prompts(classifier, prompt):
    result = classifier.classify(prompt)
    assert result.tier == ModelTier.COMPLEX, (
        f"Expected COMPLEX for '{prompt}', got score={result.final_score:.3f}"
    )


# ── Classifier: score is in valid range ───────────────────────────

def test_score_range(classifier):
    for prompt in ["Hello", "x" * 1000, "debug this ```python\ncode```"]:
        result = classifier.classify(prompt)
        assert 0.0 <= result.final_score <= 1.0


# ── Classifier: empty prompt raises ───────────────────────────────

def test_empty_prompt_raises(classifier):
    with pytest.raises(ValueError):
        classifier.classify("   ")


# ── Router: override rules ────────────────────────────────────────

def test_router_code_override(router):
    prompt = "Fix this bug:\n```python\ndef foo(): pass\n```"
    # Should be overridden to COMPLEX regardless of low score
    decision = router.route(prompt, complexity_score=0.2)
    assert decision.tier == ModelTier.COMPLEX
    assert decision.override_reason is not None


def test_router_respects_threshold(router):
    decision_simple  = router.route("Translate this.", complexity_score=0.1)
    decision_complex = router.route("Analyze this system.", complexity_score=0.9)
    assert decision_simple.tier  == ModelTier.SIMPLE
    assert decision_complex.tier == ModelTier.COMPLEX
