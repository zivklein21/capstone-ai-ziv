import asyncio
import json

from capstone_ai_ziv.core.ai_client import EXTRACT_MEETING_TOOL, AIClient, MettingSummary

transcript = """
Team stanup March 15. Alice, Bob, and  Carol attended.
We decided to switch from REST to GraphQL for the internal API.
Action items: Alice will draft the GraphQL schema by Friday.
Bob will benchmark query performance. Carol updates the documentions.
"""


async def main():
    print("=== schema ===")
    print(json.dumps(MettingSummary.model_json_schema(), indent=2))
    client = AIClient()
    try:
        summary = await client.extract_structured(transcript, EXTRACT_MEETING_TOOL, MettingSummary)
        print("=== result ===")
        print(summary.model_dump_json(indent=2))
    finally:
        await client.close()

asyncio.run(main())