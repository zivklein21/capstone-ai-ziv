import asyncio
import json
import logging

from capstone_ai_ziv.core.ai_client import AIClient

logging.basicConfig(level=logging.DEBUG)  # so the loop's trace is visible

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    }
]


async def handle_weather(params: dict) -> str:
    city = params["city"]
    # Simulated weather data
    return json.dumps({"city": city, "temp_c": 22, "condition": "sunny"})


tool_handler = {"get_weather": handle_weather}


async def main():
    client = AIClient()
    try:
        result = await client.tool_conversation(
            messages=[{"role": "user", "content": "What's the weather in Tel Aviv?"}],
            tools=TOOLS,
            tool_handler=tool_handler,
        )
        print("\n=== ANSWER ===")
        print(result)
    finally:
        await client.close()


asyncio.run(main())