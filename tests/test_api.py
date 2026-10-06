from __future__ import annotations

import base64
import math
from io import BytesIO

import httpx
import pytest
from asgi_lifespan import LifespanManager
from PIL import Image

from embeddinggemma_api.config import Settings
from embeddinggemma_api.main import create_app
from embeddinggemma_api.service import EmbeddingService


class FakeEmbeddingService(EmbeddingService):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.model = object()  # type: ignore[assignment]

    @property
    def ready(self) -> bool:
        return True

    async def encode(self, items, input_type):
        assert input_type in {"query", "document"}
        value = 1.0 / math.sqrt(768)
        return [[value] * 768 for _ in items]


def image_data_url() -> str:
    output = BytesIO()
    Image.new("RGB", (12, 7), (120, 30, 200)).save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@pytest.fixture
async def client():
    settings = Settings(model_path="/unused", model_name="embeddinggemma-2")
    application = create_app(FakeEmbeddingService(settings))
    async with LifespanManager(application):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://test"
        ) as http:
            yield http


@pytest.mark.anyio
async def test_ready_and_models(client):
    assert (await client.get("/health/live")).status_code == 200
    assert (await client.get("/health/ready")).status_code == 200
    assert (await client.get("/v1/models")).json()["data"][0]["embedding_dimension"] == 768


@pytest.mark.anyio
async def test_text_embedding(client):
    response = await client.post(
        "/v1/embeddings",
        json={"model": "embeddinggemma-2", "input": "猫", "input_type": "query"},
    )
    assert response.status_code == 200
    assert len(response.json()["data"][0]["embedding"]) == 768


@pytest.mark.anyio
async def test_image_embedding(client):
    response = await client.post(
        "/v1/embeddings",
        json={
            "model": "embeddinggemma-2",
            "input": [{"content": [{"type": "image_url", "image_url": {"url": image_data_url()}}]}],
            "input_type": "document",
        },
    )
    assert response.status_code == 200
    vector = response.json()["data"][0]["embedding"]
    assert len(vector) == 768
    assert math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0, rel_tol=0.01)


@pytest.mark.anyio
async def test_remote_image_urls_are_rejected(client):
    response = await client.post(
        "/v1/embeddings",
        json={
            "model": "embeddinggemma-2",
            "input": [
                {
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}}
                    ]
                }
            ],
        },
    )
    assert response.status_code == 400
