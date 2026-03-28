from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

from groq import Groq

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False

# ✅ IMPORTS MODELS FROM CONFIG
from config import GROQ_API_KEY, CONNECTOR, ModelTier, SIMPLE_MODEL, COMPLEX_MODEL

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    text:           str
    model_id:       str
    tier_requested: ModelTier
    tier_used:      ModelTier
    tokens_input:   int
    tokens_output:  int
    latency_ms:     float
    fallback_used:  bool = False
    error:          Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.tokens_input + self.tokens_output

class TokenCounter:
    def __init__(self):
        if _TIKTOKEN_AVAILABLE:
            self._enc = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        if _TIKTOKEN_AVAILABLE:
            return len(self._enc.encode(text))
        return max(1, int(len(text.split()) / 0.75))

class GroqClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        self.client = Groq(api_key=api_key)

    def call(self, model: str, system: str, user: str,max_tokens: int = 1024):
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.7,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

class HFConnector:
    def __init__(self):
        self._client = GroqClient(GROQ_API_KEY)
        self._tokens = TokenCounter()

    async def complete(
        self,
        prompt: str,
        tier: ModelTier,
        system_prompt: str = "You are a helpful assistant.",
    ) -> LLMResponse:

        # Pull the correct model config
        model_cfg = SIMPLE_MODEL if tier == ModelTier.SIMPLE else COMPLEX_MODEL
        model = model_cfg.model_id

        input_tokens = self._tokens.count(system_prompt) + self._tokens.count(prompt)
        t0 = time.perf_counter()

        try:
            text = await asyncio.to_thread(
                self._client.call,
                model,
                system_prompt,
                prompt,
                model_cfg.max_new_tokens # ✅ FIX: Pass the config limit to the client
            )

        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            return LLMResponse(
                text="",
                model_id=model,
                tier_requested=tier,
                tier_used=tier,
                tokens_input=input_tokens,
                tokens_output=0,
                latency_ms=latency,
                error=str(e),
            )

        latency = (time.perf_counter() - t0) * 1000
        output_tokens = self._tokens.count(text)

        return LLMResponse(
            text=text,
            model_id=model,
            tier_requested=tier,
            tier_used=tier,
            tokens_input=input_tokens,
            tokens_output=output_tokens,
            latency_ms=latency,
        )

    async def complete_with_guard(
        self,
        prompt: str,
        tier: ModelTier,
        session,
        guard,
        system_prompt: str = "You are a helpful assistant.",
    ) -> LLMResponse:

        response = await self.complete(prompt, tier, system_prompt)

        await guard.checkpoint(
            session,
            tokens_used=response.total_tokens,
            output=response.text,
        )

        return response