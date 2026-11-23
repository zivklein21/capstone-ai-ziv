# tests/unit/test_ai_client.py

class TestAIClientMessage:
    async def test_basic_message_returns_text(self, mock_ai_client):
        result = await mock_ai_client.message("Hello", system="Be brief")
        assert result == "Mocked response"


    async def test_message_passes_system_prompt(self, mock_ai_client):
        await mock_ai_client.message("Hello", system="Be concise")
        call_kwargs = mock_ai_client._client.messages.create.call_args
        assert call_kwargs.kwargs["system"] == "Be concise"


    async def test_messages_uses_configured_model(self, mock_ai_client):
        await mock_ai_client.message("Hello", system="Be brief")
        call_kwargs = mock_ai_client._client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-sonnet-5"


    async def test_messages_respects_max_tokens(self, mock_ai_client):
        await mock_ai_client.message("Hello", system="Be brief")
        call_kwargs = mock_ai_client._client.messages.create.call_args
        assert call_kwargs.kwargs["max_tokens"] == 4096


# Include this class only if you built conversation() (Stage 3, optional step 2)
class TestAIClientConversation:
    async def test_multi_turn_passes_message_history(self, mock_ai_client):
        messages = [
            {"role" : "user", "content" : "My name is Alice."},
            {"role" : "assistant", "content" : "Hello Alice!"},
            {"role" : "user", "content" : "What's my name?"},
        ]
        await mock_ai_client.conversation(messages)
        call_kwargs = mock_ai_client._client.messages.create.call_args
        assert len(call_kwargs.kwargs["messages"]) == 3
        assert call_kwargs.kwargs["messages"][0]["content"] == "My name is Alice."