"""OpenAI async client helpers.

- ``create_async_client()`` reads ``OPENAI_API_KEY`` (with optional
  ``.env`` loading) and returns an ``AsyncOpenAI`` instance.
- ``call_structured()`` wraps ``client.beta.chat.completions.parse()``
  with retry + backoff. The judge and verdict extractors use this for
  Pydantic-typed responses.
- ``call_raw()`` wraps the bare completions endpoint for the
  ``--single-turn`` calibration mode (no reasoning, ``max_tokens=2``).
- ``RateLimiter`` is an asyncio semaphore that bounds concurrent calls.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import Any, Optional, Type, TypeVar

from openai import AsyncOpenAI
from openai import APIConnectionError, APIStatusError, RateLimitError
from pydantic import BaseModel

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


def create_async_client(api_key: Optional[str] = None) -> AsyncOpenAI:
    """Return an ``AsyncOpenAI`` client.

    Picks up the API key from (in order): explicit arg, ``OPENAI_API_KEY``
    env var, ``.env`` file in repo root if ``python-dotenv`` is available.
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        try:
            from dotenv import load_dotenv  # type: ignore[import-not-found]

            load_dotenv()
            api_key = os.environ.get("OPENAI_API_KEY")
        except ImportError:
            pass
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set; export it or add to .env"
        )
    return AsyncOpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """Bounded-concurrency limiter (asyncio semaphore wrapper).

    Use as ``async with limiter: ...`` around each API call.
    """

    def __init__(self, concurrency: int):
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self._sem = asyncio.Semaphore(concurrency)
        self.concurrency = concurrency

    async def __aenter__(self) -> "RateLimiter":
        await self._sem.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._sem.release()


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

_RETRYABLE = (RateLimitError, APIConnectionError)


async def _with_retry(
    coro_factory,
    *,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> Any:
    """Call ``await coro_factory()`` with exponential-backoff retry.

    Retries on rate-limit / connection errors and on ``APIStatusError``
    with 5xx status codes. Other exceptions surface immediately.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return await coro_factory()
        except _RETRYABLE as e:
            if attempt >= max_attempts:
                log.error("giving up after %d attempts: %s", attempt, e)
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay *= 0.5 + random.random()  # jitter
            log.warning("retryable error %s; sleeping %.1fs (attempt %d/%d)", type(e).__name__, delay, attempt, max_attempts)
            await asyncio.sleep(delay)
        except APIStatusError as e:
            if 500 <= e.status_code < 600 and attempt < max_attempts:
                delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                delay *= 0.5 + random.random()
                log.warning("server error %d; sleeping %.1fs (attempt %d/%d)", e.status_code, delay, attempt, max_attempts)
                await asyncio.sleep(delay)
                continue
            raise


# ---------------------------------------------------------------------------
# Structured-output call
# ---------------------------------------------------------------------------


async def call_structured(
    client: AsyncOpenAI,
    *,
    model: str,
    messages: list[dict],
    response_format: Type[T],
    temperature: float = 0.0,
    seed: int = 42,
    max_attempts: int = 5,
) -> T:
    """Single structured-output call returning a parsed Pydantic model.

    Uses ``client.beta.chat.completions.parse`` and the structured-output
    feature so the model is forced to emit a JSON object matching
    ``response_format``.
    """

    async def _do() -> T:
        completion = await client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
            seed=seed,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError(
                f"structured-output parse returned None for model={model}"
            )
        return parsed

    return await _with_retry(_do, max_attempts=max_attempts)


async def call_raw(
    client: AsyncOpenAI,
    *,
    model: str,
    messages: list[dict],
    max_tokens: int = 2,
    temperature: float = 0.0,
    seed: int = 42,
    max_attempts: int = 5,
) -> str:
    """Bare completion (no structured output). Used for single-turn calibration.

    Returns the raw assistant text, stripped.
    """

    async def _do() -> str:
        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        )
        return (completion.choices[0].message.content or "").strip()

    return await _with_retry(_do, max_attempts=max_attempts)
