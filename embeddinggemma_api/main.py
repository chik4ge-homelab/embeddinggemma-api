from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import Settings
from .service import EmbeddingService, parse_input

LOGGER = logging.getLogger("embeddinggemma_api")


def _log(event: str, **fields: Any) -> None:
    LOGGER.info(json.dumps({"event": event, **fields}, separators=(",", ":"), sort_keys=True))


def _request_id(request: Request) -> str:
    return request.headers.get("x-request-id", "") or "unknown"


def create_app(service: EmbeddingService | None = None) -> FastAPI:
    settings = service.settings if service is not None else Settings.from_env()
    embedding_service = service or EmbeddingService(settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if embedding_service.model is None:
            try:
                await __import__("asyncio").to_thread(embedding_service.load)
            except Exception as error:  # noqa: BLE001
                embedding_service.load_error = type(error).__name__
                LOGGER.exception("EmbeddingGemma model load failed")
        application.state.embedding_service = embedding_service
        _log(
            "startup",
            model=settings.model_name,
            model_repo=settings.model_hub_repo,
            model_revision=settings.model_revision,
            ready=embedding_service.ready,
        )
        yield

    application = FastAPI(title="EmbeddingGemma API", lifespan=lifespan)

    @application.get("/")
    async def root() -> dict[str, str]:
        return {"message": "EmbeddingGemma API"}

    @application.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/health/ready")
    async def ready(request: Request) -> JSONResponse:
        current = request.app.state.embedding_service
        if not current.ready:
            return JSONResponse(
                {"status": "not_ready", "reason": current.load_error or "model_not_loaded"},
                status_code=503,
            )
        return JSONResponse({"status": "ready", "dimension": 768})

    @application.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": settings.model_name,
                    "object": "model",
                    "owned_by": "google",
                    "embedding_dimension": 768,
                }
            ],
        }

    @application.post("/v1/embeddings")
    async def embeddings(request: Request) -> JSONResponse:
        request_id = _request_id(request)
        started = time.monotonic()
        current: EmbeddingService = request.app.state.embedding_service
        try:
            if not current.ready:
                raise RuntimeError("model is not ready")
            body = await request.json()
            if not isinstance(body, dict) or body.get("model") != settings.model_name:
                raise ValueError("model must be embeddinggemma-2")
            items = parse_input(body.get("input"), settings.max_image_bytes)
            input_type = body.get("input_type", "document")
            vectors = await current.encode(items, input_type)
            result = {
                "object": "list",
                "model": settings.model_name,
                "data": [
                    {"object": "embedding", "index": index, "embedding": vector}
                    for index, vector in enumerate(vectors)
                ],
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            }
            modality = "mixed" if len({item.kind for item in items}) > 1 else items[0].kind
            _log(
                "request_complete",
                request_id=request_id,
                endpoint="/v1/embeddings",
                model=settings.model_name,
                modality=modality,
                batch_size=len(items),
                inference_latency_ms=round((time.monotonic() - started) * 1000, 2),
                total_latency_ms=round((time.monotonic() - started) * 1000, 2),
                http_status=200,
            )
            return JSONResponse(result)
        except ValueError as error:
            _log("request_error", request_id=request_id, endpoint="/v1/embeddings", http_status=400)
            return JSONResponse(
                {
                    "error": {"type": "invalid_request_error", "message": str(error)},
                    "request_id": request_id,
                },
                status_code=400,
            )
        except RuntimeError as error:
            _log("request_error", request_id=request_id, endpoint="/v1/embeddings", http_status=503)
            return JSONResponse(
                {
                    "error": {"type": "model_not_ready", "message": str(error)},
                    "request_id": request_id,
                },
                status_code=503,
            )
        except Exception:  # noqa: BLE001
            LOGGER.exception("embedding request failed")
            return JSONResponse(
                {
                    "error": {"type": "server_error", "message": "embedding request failed"},
                    "request_id": request_id,
                },
                status_code=500,
            )

    return application


app = create_app()
