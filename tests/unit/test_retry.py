# tests/unit/test_retry.py

import httpx
import pytest
from unittest.mock import AsyncMock, patch

from capstone_ai_ziv.core.retry import with_retry


def make_response(status_code: int) -> httpx.Response:
    # raise_for_status() requiers the originating request to be attached.
    return httpx.Response(status_code, request=httpx.Request("GET", "https://test"))


class TestRetryLogic:
    async def test_succeeds_on_first_try(self):
        fn = AsyncMock(return_value=make_response(200))
        response = await with_retry(fn, max_retries=3, base_delay=0.01)
        assert response.status_code == 200
        assert fn.call_count == 1


    async def test_retries_transient_errors_then_succeeds(self):
        fn = AsyncMock(
            side_effect=[
                httpx.ConnectTimeout("boom"),
                httpx.ConnectTimeout("boom"),
                make_response(200),
            ]
        )
        response = await with_retry(fn, max_retries=3, base_delay=0.01)
        assert response.status_code == 200
        assert fn.call_count == 3


    async def test_retries_after_max_retries(self):
        fn = AsyncMock(return_value=make_response(503))
        with pytest.raises(httpx.HTTPStatusError):
            await with_retry(fn, max_retries=2, base_delay=0.01)
        assert fn.call_count == 3


    async def test_does_not_retry_non_retryable_status(self):
        fn = AsyncMock(return_value=make_response(404))
        with pytest.raises(httpx.HTTPStatusError):
            await with_retry(fn, max_retries=2, base_delay=0.01)
        assert fn.call_count == 1 # a 404 is not transient - no retries


    async def test_does_not_retry_programming_errors(self):
        # A TypeError in your own code must surface immediately, not 4x slower.
        fn = AsyncMock(side_effect=TypeError("bug in my code"))
        with pytest.raises(TypeError):
            await with_retry(fn, max_retries=3, base_delay=0.01)
        assert fn.call_count == 1


    async def test_exponential_backoff_doubles_the_delay(self):
        fn = AsyncMock(
            side_effect=[
                httpx.ConnectTimeout("boom"),
                httpx.ConnectTimeout("boom"),
                make_response(200),
            ]
        )
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await with_retry(fn, max_retries=3, base_delay=0.01)
        delays = [c.args[0] for c in mock_sleep.await_args_list] 
        assert delays == [0.01, 0.02]


