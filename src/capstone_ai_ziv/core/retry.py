# src/capstone_ai_ziv/core/retry.py

import asyncio
import logging
from collections.abc import Awaitable, Callable
import httpx

logger = logging.getLogger(__name__)


def _retry_after(response: httpx.Response, fallback: float) -> float:
    """A 429 often tells you excatly how lont to wait. Honour it when it does."""
    header = response.headers.get("retry-after", "")
    return float(header) if header.isdigit() else fallback


async def with_retry(
        fn: Callable[..., Awaitable[httpx.Response]],
        *args,
        max_retries: int = 3,
        base_delay: float = 1.0,
        retryable_status: tuple[int, ...] = (429, 500, 502, 503),
        **kwargs,
) -> httpx.Response:
    for attempt  in range (max_retries + 1):
        try:
            response = await fn(*args, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in retryable_status:
                raise
            if attempt == max_retries:
                raise
            delay = _retry_after(exc.response, base_delay * 2**attempt)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if attempt == max_retries:
                raise
            delay = base_delay * 2**attempt
        logger.warning(
            "Attempt %d/%d failed, retrying in %.1fs",
            attempt + 1, max_retries + 1, delay,
        )
        await asyncio.sleep(delay)
    raise RuntimeError("unreachable")