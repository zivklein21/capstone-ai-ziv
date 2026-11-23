# tests/unit/conftest.py

import pytest
from unittest.mock import AsyncMock, MagicMock
from httpx import ASGITransport, AsyncClient

from capstone_ai_ziv.core.ai_client import AIClient
from capstone_ai_ziv.api.main import app


@pytest.fixture
def mock_anthropic_response():
    # type="text" is load-bearing: a bare MagicMock answers .type with another
    # MagicMock - truthy, but never equal to "text" - so any code that selects
    # blocks by type would silently find nothing and return an empty string.
    response = MagicMock()
    response.content = [MagicMock(text="Mocked response", type="text")]
    response.stop_reason = "end_turn"
    response.usage = MagicMock(
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    return response


@pytest.fixture
def mock_ai_client(mock_anthropic_response):
    # __new__ skips __init__, so  no API key is needed and no one real SDK client
    # is constructed. Every attribute the methods touch is set explicity.
    client = AIClient.__new__(AIClient)
    client._client = AsyncMock()
    client._model = "claude-sonnet-5"
    client._max_tokens = 4096
    client._client.messages.create = AsyncMock(return_value=mock_anthropic_response)
    return client


@pytest.fixture
async def api_client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
