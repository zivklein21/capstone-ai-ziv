# src/capstone_ai_ziv/api/main.py

import logging
import time
import uuid

from anthropic import APIError

import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import APIKeyHeader

from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from capstone_ai_ziv.core.ai_client import AIClient
from capstone_ai_ziv.core.config import settings
from capstone_ai_ziv.api.models import ChatRequest, ChatResponse, ToolRequest, ErrorResponse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ai_client = AIClient()
    yield
    await app.state.ai_client.close()


app = FastAPI(title="Capstone AI Ziv API", version="0.1.0", lifespan=lifespan)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()

        response = await call_next(request)

        elapsed =time.perf_counter() - start
        logger.info(
            "[%s] %s %s -> %d (%.3fs)",
            request_id, request.method, request.url.path,
            response.status_code, elapsed,
        )
        response.headers["X-Request-ID"] = request_id
        return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)


@app.exception_handler(APIError)
async def anthropic_error_handler(request: Request, exc: APIError):
    return JSONResponse(
        status_code=502,
        content=ErrorResponse(error="AI service error", detail=str(exc)).model_dump(),
    )


API_KEY_HEADER =APIKeyHeader(name="X-API-KEY", auto_error=True)


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    excepted = "dev-key-change-in-production"
    if api_key != excepted:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


@app.get("/health")
async def health():
    return {"status" : "ok"}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat(request: ChatRequest):
    try:
        result = await app.state.ai_client.message(
            request.message, system=request.system
        )
        return ChatResponse(response=result, model=settings.model_name)
    except APIError:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/chat/stream", dependencies=[Depends(verify_api_key)])
async def chat_stream(request: ChatRequest):
    async def event_generator():
        try:
            async for chunk in app.state.ai_client.stream_message(
                request.message, system=request.system
            ):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            yield f"data: [DEBUG] {exc}\n\n"
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current wether for a city. Use this when the user asks about weather.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"types": "string", "description": "City name"},
            },
            "requierd": ["city"],
        },
    }
]


async def handle_weather(params: dict) -> str:
    return json.dumps({"city": params["city"], "temp_c": 22, "condition": "sunny"})


TOOL_HANDLERS = {"get_weather": handle_weather}


@app.post("/chat/tools", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat_with_tools(request: ToolRequest):
    try:
        result = await app.state.ai_client.tool_conversation(
            messages=[{"role": "user", "content": request.message}],
            tools = TOOLS,
            tool_handler=TOOL_HANDLERS,
            system=request.system,
        )
        return ChatResponse(response=result, model=settings.model_name)
    except APIError:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))