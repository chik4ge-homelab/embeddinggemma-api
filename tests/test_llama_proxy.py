from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager

from embeddinggemma_api.config import Settings
from embeddinggemma_api.llama_proxy import create_app

RealAsyncClient = httpx.AsyncClient
IMAGE_URL = "http://image-backend:8081"
TEXT_URL = "http://text-backend:8081"


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any] | None = None):
        self.status_code = status_code
        self.payload = payload or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "backend error",
                request=httpx.Request("POST", "http://backend"),
                response=httpx.Response(self.status_code),
            )

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeBackendClient:
    def __init__(self, text_health_status: int = 200):
        self.text_health_status = text_health_status
        self.posts: list[tuple[str, dict[str, Any]]] = []

    async def get(self, url: str) -> FakeResponse:
        status = self.text_health_status if url.startswith(TEXT_URL) else 200
        return FakeResponse(status)

    async def post(self, url: str, json: dict[str, Any]) -> FakeResponse:
        self.posts.append((url, json))
        vector = [1.0] * 768
        return FakeResponse(
            200,
            {"data": [{"embedding": vector} for _ in json["input"]]},
        )

    async def aclose(self) -> None:
        pass


def make_app(monkeypatch, text_health_status: int = 200, split_backends: bool = True):
    fake_client = FakeBackendClient(text_health_status)
    monkeypatch.setenv("LLAMA_IMAGE_SERVER_URL", IMAGE_URL)
    monkeypatch.setenv("LLAMA_DEVICE", "SYCL0")
    if split_backends:
        monkeypatch.setenv("LLAMA_TEXT_SERVER_URL", TEXT_URL)
        monkeypatch.setenv("LLAMA_TEXT_DEVICE", "CPU")
    else:
        monkeypatch.delenv("LLAMA_TEXT_SERVER_URL", raising=False)
        monkeypatch.delenv("LLAMA_TEXT_DEVICE", raising=False)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_: fake_client)
    settings = Settings(model_path="/unused", model_name="embeddinggemma-2")
    return create_app(settings), fake_client


async def wait_for_image_warmup(application) -> None:
    for _ in range(200):
        if application.state.image_warmup_complete:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("image backend warmup did not complete")


@pytest.mark.anyio
async def test_text_embeddings_use_text_backend_and_keep_task_prefix(monkeypatch):
    application, backend = make_app(monkeypatch)
    async with LifespanManager(application):
        await wait_for_image_warmup(application)
        async with RealAsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/embeddings",
                json={"model": "embeddinggemma-2", "input": "猫", "input_type": "query"},
            )

    assert response.status_code == 200
    request = next(body for url, body in backend.posts if url == f"{TEXT_URL}/v1/embeddings")
    assert request["input"] == ["task: search result | query: 猫"]
    assert len(response.json()["data"][0]["embedding"]) == 768


@pytest.mark.anyio
async def test_single_backend_configuration_stays_on_image_server(monkeypatch):
    application, backend = make_app(monkeypatch, split_backends=False)
    async with LifespanManager(application):
        await wait_for_image_warmup(application)
        async with RealAsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/embeddings",
                json={"model": "embeddinggemma-2", "input": "猫", "input_type": "query"},
            )
            readiness = await client.get("/health/ready")

    assert response.status_code == 200
    assert any(url == f"{IMAGE_URL}/v1/embeddings" for url, _ in backend.posts)
    assert readiness.json()["text_device"] == "SYCL0"


@pytest.mark.anyio
async def test_image_and_mixed_embeddings_use_image_backend(monkeypatch):
    application, backend = make_app(monkeypatch)
    image = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,cG5n"},
    }
    async with LifespanManager(application):
        await wait_for_image_warmup(application)
        async with RealAsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/embeddings",
                json={
                    "model": "embeddinggemma-2",
                    "input": ["caption", image],
                    "input_type": "document",
                },
            )

    assert response.status_code == 200
    request = next(
        body
        for url, body in backend.posts
        if url == f"{IMAGE_URL}/v1/embeddings" and len(body["input"]) == 2
    )
    assert request["input"][0] == "title: none | text: caption"
    assert request["input"][1] == {"content": [image]}


@pytest.mark.anyio
async def test_ready_requires_text_backend_health(monkeypatch):
    application, _ = make_app(monkeypatch, text_health_status=503)
    async with LifespanManager(application):
        await wait_for_image_warmup(application)
        async with RealAsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://test"
        ) as client:
            response = await client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["reason"] == "backend_health"
