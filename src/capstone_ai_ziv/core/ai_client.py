
import json
import logging
import anthropic
from collections.abc import AsyncIterator

import anthropic
from pydantic import BaseModel, Field
from capstone_ai_ziv.core.config import settings
from capstone_ai_ziv.core.prompt_loader import load_prompt

logger = logging.getLogger(__name__)


def first_text(content: list) -> str:
    """Return the first text block's text, skipping thinking and tool_use blocks."""
    for block in content:
        if block.type == "text":
            return block.text
    return ""


class MettingSummary(BaseModel):
    title: str = Field(description="Short meeting title, e.g. 'Q3 Planning Standup'")
    date: str = Field(description="Meeting date in ISO format (YYYT-MM-DD)")
    attednees: list[str] = Field(description="Full names of people who attended")
    action_items: list[str] = Field(description="One entry per commitment, prefixed with the owner's name")
    key_decisions: list[str] = Field(description="Decisions reached, excluding items still under discussion")


EXTRACT_MEETING_TOOL = {
    "name" : "extract_meeting_data",
    "description" : "Extract structured meeting data from text. Always use this tool to return meeting information.",
    "input_schema" : MettingSummary.model_json_schema(),
}

class AIClient:
    def __init__(self):
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model = settings.model_name
        self._max_tokens = settings.max_tokens

    async def message(self, user_message: str, system: str = "") -> str:
        if not system:
            system = load_prompt("system_default.txt")
        params: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": user_message}],
            "system": system,
        }
        response = await self._client.messages.create(**params)
        return first_text(response.content)

    async def conversation(
        self,
        messages: list[dict],
        system: str = "",
    ) -> anthropic.types.Message:
        params: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
        }
        if system:
            params["system"] = system
        return await self._client.messages.create(**params)

    async def stream_message(
            self,
            user_message: str,
            system: str = "",
    ) -> AsyncIterator[str]:
        params: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role" : "user", "content" : user_message}],
        }
        if system:
            params["system"] = system

        async with self._client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield text

    async def tool_conversation(
            self,
            messages: list[dict],
            tools: list[dict],
            tool_handler: dict,
            system: str = "",
            max_tool_rounds: int = 5,
    ) -> str:
        params: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": list(messages),
            "tools": tools,
        }
        if system:
            params["system"] = system

        for round_num in range(1, max_tool_rounds + 1):
            response = await self._client.messages.create(**params)
            logger.debug("tool round %d: stop_reason=%s", round_num, response.stop_reason)

            # Loop ONLY on tool_uses. Every other stop_reason is terminal.
            if response.stop_reason != "tool_use":
                text_blocks = [b.text for b in response.content if b.type == "text"]
                return "\n".join(text_blocks)

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    handler = tool_handler.get(block.name)
                    if handler:
                        result = await handler(block.input)
                    else:
                        result = f"Error: unknown tool `{block.name}`"
                    logger.debug("tool %s(%s) -> %s", block.name, json.dumps(block.input), result)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    })
            params["messages"].append({"role" : "assistant", "content": response.content})
            params["messages"].append({"role" : "user", "content": tool_results})

        return "Max tool round exceeded"

    async def extract_structured(
        self,
        text: str,
        tool_schema: dict,
        model_class: type[BaseModel],
    ) -> BaseModel:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            tools=[tool_schema],
            tool_choice={"type": "tool", "name": tool_schema["name"]},
            messages=[{"role": "user", "content": text}],
        )
        for block in response.content:
            if block.type == "tool_use":
                return model_class.model_validate(block.input)
        raise ValueError("No tool use block in response")
    
    async def close(self):
        await self._client.close()