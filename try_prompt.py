# try_prompt.py

import asyncio

from capstone_ai_ziv.core.ai_client import AIClient
from capstone_ai_ziv.core.prompt_loader import load_prompt


async def main():
    print("=== default prompt ===")
    print(load_prompt("system_default.txt"))
    client = AIClient()
    try:
        result = await client.message("What is a Python list?")
        print("=== answer (no system passed) ===")
        print(result)
    finally:
        await client.close()

asyncio.run(main())