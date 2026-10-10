from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import Settings

LOGGER = logging.getLogger("embeddinggemma_api")
QUERY_PREFIX = "task: search result | query: "
DOCUMENT_PREFIX = "title: none | text: "


def _log(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, separators=(",", ":"), sort_keys=True), flush=True)


def _image_block(item: Any, max_image_bytes: int) -> dict[str, Any]:
    if isinstance(item, dict) and "content" in item:
        content = item["content"]
    elif isinstance(item, dict) and item.get("type") == "image_url":
        content = [item]
    else:
        raise ValueError("input items must be strings or image content blocks")

    if not isinstance(content, list) or len(content) != 1 or not isinstance(content[0], dict):
        raise ValueError("content must contain exactly one image block")
    block = content[0]
    if block.get("type") != "image_url":
        raise ValueError("only image_url content blocks are supported")
    image_url = block.get("image_url")
    if not isinstance(image_url, dict) or not isinstance(image_url.get("url"), str):
        raise ValueError("image_url.url is required")
    header, separator, encoded = image_url["url"].partition(",")
    if not separator or not header.startswith("data:image/") or ";base64" not in header:
        raise ValueError("image_url must be a base64 image data URL")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise ValueError("image_url contains invalid base64") from error
    if not image_bytes or len(image_bytes) > max_image_bytes:
        raise ValueError("image_url exceeds the configured size limit")
    return {"content": [block]}


def _prepare_input(value: Any, input_type: str, max_image_bytes: int) -> list[str | dict[str, Any]]:
    if input_type not in {"query", "document"}:
        raise ValueError("input_type must be query or document")
    values = value if isinstance(value, list) else [value]
    if not values:
        raise ValueError("input must not be empty")

    prefix = QUERY_PREFIX if input_type == "query" else DOCUMENT_PREFIX
    prepared: list[str | dict[str, Any]] = []
    for item in values:
        if isinstance(item, str):
            prepared.append(prefix + item)
        else:
            prepared.append(_image_block(item, max_image_bytes))
    return prepared


def _normalized_embeddings(payload: dict[str, Any], expected_count: int) -> list[list[float]]:
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != expected_count:
        raise RuntimeError("llama.cpp returned an unexpected embedding batch")

    vectors: list[list[float]] = []
    for item in data:
        if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
            raise RuntimeError("llama.cpp returned a malformed embedding")
        vector = [float(value) for value in item["embedding"]]
        if len(vector) != 768 or not all(math.isfinite(value) for value in vector):
            raise RuntimeError("llama.cpp returned an invalid embedding vector")
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isfinite(norm) or norm == 0:
            raise RuntimeError("llama.cpp returned an unnormalizable embedding")
        vectors.append([value / norm for value in vector])
    return vectors


def create_app(settings: Settings | None = None) -> FastAPI:
    current_settings = settings or Settings.from_env()
    server_url = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8081").rstrip("/")
    device = os.environ.get("LLAMA_DEVICE", "SYCL0")
    semaphore = asyncio.Semaphore(max(1, current_settings.max_concurrency))

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.llama_client = httpx.AsyncClient(
            timeout=httpx.Timeout(600.0, connect=10.0)
        )
        yield
        await application.state.llama_client.aclose()

    application = FastAPI(title="EmbeddingGemma API", lifespan=lifespan)

    @application.get("/")
    async def root() -> dict[str, str]:
        return {"message": "EmbeddingGemma API"}

    @application.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/health/ready")
    async def ready(request: Request) -> JSONResponse:
        try:
            response = await request.app.state.llama_client.get(f"{server_url}/health")
        except httpx.HTTPError:
            response = None
        if response is None or response.status_code != 200:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return JSONResponse({"status": "ready", "dimension": 768, "device": device})

    @application.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": current_settings.model_name,
                    "object": "model",
                    "owned_by": "google",
                    "embedding_dimension": 768,
                }
            ],
        }

    @application.post("/v1/embeddings")
    async def embeddings(request: Request) -> JSONResponse:
        request_id = request.headers.get("x-request-id", "") or "unknown"
        started = time.monotonic()
        try:
            body = await request.json()
            if not isinstance(body, dict) or body.get("model") != current_settings.model_name:
                raise ValueError(f"model must be {current_settings.model_name}")
            items = _prepare_input(
                body.get("input"),
                body.get("input_type", "document"),
                current_settings.max_image_bytes,
            )
        except ValueError as error:
            return JSONResponse(
                {"error": {"type": "invalid_request_error", "message": str(error)}},
                status_code=400,
            )

        queued_at = time.monotonic()
        async with semaphore:
            inference_started = time.monotonic()
            try:
                response = await request.app.state.llama_client.post(
                    f"{server_url}/v1/embeddings",
                    json={"model": current_settings.model_name, "input": items},
                )
                response.raise_for_status()
                payload = response.json()
                vectors = _normalized_embeddings(payload, len(items))
            except httpx.HTTPError as error:
                _log(
                    "request_error",
                    request_id=request_id,
                    endpoint="/v1/embeddings",
                    error_type=type(error).__name__,
                    http_status=503,
                )
                return JSONResponse(
                    {"error": {"type": "backend_unavailable", "message": "llama.cpp request failed"}},
                    status_code=503,
                )
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                LOGGER.exception("llama.cpp returned an invalid embedding response")
                _log(
                    "request_error",
                    request_id=request_id,
                    endpoint="/v1/embeddings",
                    error_type=type(error).__name__,
                    http_status=502,
                )
                return JSONResponse(
                    {"error": {"type": "backend_error", "message": "invalid llama.cpp response"}},
                    status_code=502,
                )

        elapsed_ms = round((time.monotonic() - started) * 1000, 2)
        result = {
            "object": "list",
            "model": current_settings.model_name,
            "data": [
                {"object": "embedding", "index": index, "embedding": vector}
                for index, vector in enumerate(vectors)
            ],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }
        has_text = any(isinstance(item, str) for item in items)
        has_image = any(isinstance(item, dict) for item in items)
        _log(
            "request_complete",
            request_id=request_id,
            endpoint="/v1/embeddings",
            model=current_settings.model_name,
            modality="mixed" if has_text and has_image else "image" if has_image else "text",
            batch_size=len(items),
            device=device,
            queue_wait_ms=round((inference_started - queued_at) * 1000, 2),
            inference_ms=round((time.monotonic() - inference_started) * 1000, 2),
            total_ms=elapsed_ms,
            inference_latency_ms=round((time.monotonic() - inference_started) * 1000, 2),
            total_latency_ms=elapsed_ms,
            http_status=200,
        )
        return JSONResponse(result)

    return application
