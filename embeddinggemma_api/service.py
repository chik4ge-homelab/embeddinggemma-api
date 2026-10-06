from __future__ import annotations

import asyncio
import base64
import os
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sentence_transformers import SentenceTransformer

from .config import Settings


@dataclass(frozen=True)
class InputItem:
    kind: Literal["text", "image"]
    value: str | Image.Image


def _image_from_data_url(url: str, max_bytes: int) -> Image.Image:
    header, separator, encoded = url.partition(",")
    if not separator or not header.startswith("data:image/") or ";base64" not in header:
        raise ValueError("image_url must be a base64 image data URL")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise ValueError("image_url contains invalid base64") from error
    if not image_bytes or len(image_bytes) > max_bytes:
        raise ValueError("image_url exceeds the configured size limit")
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("image_url is not a readable image") from error


def _parse_content(content: Any, max_image_bytes: int) -> InputItem:
    if not isinstance(content, list) or len(content) != 1 or not isinstance(content[0], dict):
        raise ValueError("content must contain exactly one image block")
    block = content[0]
    if block.get("type") != "image_url":
        raise ValueError("only image_url content blocks are supported")
    image_url = block.get("image_url")
    if not isinstance(image_url, dict) or not isinstance(image_url.get("url"), str):
        raise ValueError("image_url.url is required")
    return InputItem("image", _image_from_data_url(image_url["url"], max_image_bytes))


def parse_input(value: Any, max_image_bytes: int) -> list[InputItem]:
    values = value if isinstance(value, list) else [value]
    if not values:
        raise ValueError("input must not be empty")
    result: list[InputItem] = []
    for item in values:
        if isinstance(item, str):
            result.append(InputItem("text", item))
        elif isinstance(item, dict) and "content" in item:
            result.append(_parse_content(item["content"], max_image_bytes))
        elif isinstance(item, dict) and item.get("type") == "image_url":
            result.append(_parse_content([item], max_image_bytes))
        else:
            raise ValueError("input items must be strings or image content blocks")
    return result


class EmbeddingService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model: SentenceTransformer | None = None
        self.load_error: str | None = None
        self.loaded_at: float | None = None
        self._semaphore = asyncio.Semaphore(max(1, settings.max_concurrency))

    def load(self) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        torch.set_num_threads(self.settings.torch_threads)
        try:
            torch.set_num_interop_threads(max(1, min(self.settings.torch_threads, 4)))
        except RuntimeError:
            pass
        model = SentenceTransformer(
            self.settings.model_path,
            config_kwargs={"audio_config": None},
            model_kwargs={"torch_dtype": torch.float32},
            device="cpu",
        )
        dimension = model.get_sentence_embedding_dimension()
        if dimension != 768:
            raise RuntimeError(f"unexpected model dimension: {dimension}")
        self.model = model
        self.load_error = None
        self.loaded_at = time.monotonic()

    @property
    def ready(self) -> bool:
        return self.model is not None and self.load_error is None

    def _encode_sync(self, items: list[InputItem], input_type: str) -> list[list[float]]:
        if self.model is None:
            raise RuntimeError("model is not loaded")
        if input_type not in {"query", "document"}:
            raise ValueError("input_type must be query or document")

        if all(item.kind == "text" for item in items):
            values = [item.value for item in items]
            prompt_name = "SearchQuery" if input_type == "query" else "Document"
            encoded = self.model.encode(
                values,
                prompt_name=prompt_name,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        elif all(item.kind == "image" for item in items):
            encoded = self.model.encode(
                [item.value for item in items],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            encoded = np.asarray(
                [
                    self.model.encode(
                        item.value,
                        prompt_name=("SearchQuery" if input_type == "query" else "Document")
                        if item.kind == "text"
                        else None,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                    for item in items
                ]
            )

        matrix = np.asarray(encoded, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape != (len(items), 768):
            raise RuntimeError(f"unexpected embedding shape: {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise RuntimeError("model returned non-finite embedding values")
        norms = np.linalg.norm(matrix, axis=1)
        if not np.all(np.isfinite(norms)) or not np.all(np.abs(norms - 1.0) <= 0.01):
            raise RuntimeError("model returned an unnormalized embedding")
        return matrix.tolist()

    async def encode(
        self, items: list[InputItem], input_type: str
    ) -> tuple[list[list[float]], float, float]:
        queued_at = time.monotonic()
        async with self._semaphore:
            inference_started = time.monotonic()
            vectors = await asyncio.to_thread(self._encode_sync, items, input_type)
        return (
            vectors,
            round((inference_started - queued_at) * 1000, 2),
            round((time.monotonic() - inference_started) * 1000, 2),
        )
