# tests/integration/test_api.py

from unittest.mock import AsyncMock, patch

from capstone_ai_ziv.api.main import app
from capstone_ai_ziv.core.config import settings

AUTH = {"X-API-Key" : "dev-key-change-in-production"}


class TestHealthEndpoint:
    async def test_health_returns_ok(self, api_client):
        response = await api_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status":"ok"}


class TestChatEndpoint:
    async def test_chat_returns_response(self, api_client):
        with patch.object(app.state, "ai_client", create=True) as mock_client:
            mock_client.message = AsyncMock(return_value="Test response")
            response = await api_client.post(
                "/chat", json={"message": "Hello"}, headers=AUTH
            )
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Test response"
        assert data["model"] == settings.model_name


    async def test_chat_rejects_empty_messages(self, api_client):
        response = await api_client.post(
            "/chat", json={"message": ""}, headers=AUTH
        )
        assert response.status_code == 422 # Pydantic validation error


    async def test_chat_rejects_missing_api_key(self, api_client):
        response = await api_client.post("/chat", json={"message":"Hello"})
        assert response.status_code == 401


class TestStreamEndpoint:
    async def test_stream_returns_sse(self, api_client):
        async def mock_stream(msg, system=""):
            for chunk in ["Hello", " world", "!"]:
                yield chunk

        with patch.object(app.state, "ai_client", create=True) as mock_client:
            mock_client.stream_message = mock_stream
            response = await api_client.post(
                "/chat/stream", json={"message": "Hello"}, headers=AUTH
            )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        body = response.text
        assert "data: Hello" in body
        assert "[DONE]" in body


class TestToolsEndpoint:
    async def test_tools_returns_synthesized_response(self, api_client):
        with patch.object(app.state, "ai_client", create=True) as mock_client:
            mock_client.tool_conversation = AsyncMock(
                return_value = "It is 22C and sunny in Paris."
            )
            response = await api_client.post(
                "/chat/tools",
                json={"message": "What's the weather in Paris?"},
                headers=AUTH,
            )
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "It is 22C and sunny in Paris."
        assert data["model"] == settings.model_name

    