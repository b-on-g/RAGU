"""LLM wrapper that rotates API keys across multiple CachedAsyncOpenAI clients.

Keys are read from LLM_API_KEYS env var (comma-separated) or fall back to LLM_API_KEY.
"""

import asyncio
import os
from ragu.models import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI


def _load_keys() -> list[str]:
    """Load API keys from LLM_API_KEYS (comma-separated) or LLM_API_KEY."""
    multi = os.getenv("LLM_API_KEYS", "")
    if multi:
        return [k.strip() for k in multi.split(",") if k.strip()]
    single = os.getenv("LLM_API_KEY", "")
    return [single] if single else []


class RotatingLLM:
    """Distributes LLM calls round-robin across multiple API keys.

    Each key gets its own CachedAsyncOpenAI with independent rate limiter.
    Batches are processed in small windows to avoid overwhelming the API.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        keys: list[str] | None = None,
        rpm_per_key: int = 5,
    ):
        keys = keys or _load_keys()
        if not keys:
            raise ValueError("No API keys found. Set LLM_API_KEYS or LLM_API_KEY env var.")

        self._llms: list[LLMOpenAI] = []
        for key in keys:
            client = CachedAsyncOpenAI(
                base_url=base_url,
                api_key=key,
                rate_max_per_minute=rpm_per_key,
                rate_max_simultaneous=1,
                rate_min_delay=3,
                retry_times_sec=(30, 60, 120),
            )
            self._llms.append(LLMOpenAI(client=client, model_name=model_name))

        self._idx = 0
        self._batch_size = min(5, len(self._llms))
        print(f"[RotatingLLM] {len(self._llms)} keys, {rpm_per_key} RPM each, batch_size={self._batch_size}")

    def _next(self) -> LLMOpenAI:
        llm = self._llms[self._idx % len(self._llms)]
        self._idx += 1
        return llm

    async def chat_completion(self, conversation, output_schema=None, **kwargs):
        return await self._next().chat_completion(
            conversation=conversation,
            output_schema=output_schema,
            **kwargs,
        )

    async def batch_chat_completion(self, conversations, output_schema=None, desc=None, **kwargs):
        """Process conversations in small batches to stay within rate limits."""
        results = []
        total = len(conversations)
        batch_size = self._batch_size

        for start in range(0, total, batch_size):
            batch = conversations[start:start + batch_size]
            tasks = []
            for i, conv in enumerate(batch):
                llm = self._llms[(start + i) % len(self._llms)]
                tasks.append(
                    llm.chat_completion(
                        conversation=conv,
                        output_schema=output_schema,
                        **kwargs,
                    )
                )
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for j, res in enumerate(batch_results):
                if isinstance(res, Exception):
                    print(f"[RotatingLLM] Retrying failed request ({type(res).__name__}), waiting 60s...")
                    await asyncio.sleep(60)
                    try:
                        res = await self._llms[(start + j) % len(self._llms)].chat_completion(
                            conversation=batch[j],
                            output_schema=output_schema,
                            **kwargs,
                        )
                    except Exception as e2:
                        print(f"[RotatingLLM] Retry also failed: {e2}, skipping")
                        res = None
                results.append(res)

            done = min(start + batch_size, total)
            if desc:
                print(f"  {desc}: {done}/{total}")

            if start + batch_size < total:
                await asyncio.sleep(15)

        return results
