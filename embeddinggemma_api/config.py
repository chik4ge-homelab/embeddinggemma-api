import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


@dataclass(frozen=True)
class Settings:
    model_path: str = "/models/embeddinggemma-2"
    model_name: str = "embeddinggemma-2"
    model_hub_repo: str = "google/embeddinggemma-2"
    model_revision: str = "914f7f89142e33e77833254d9c9b90c3cef7303b"
    torch_threads: int = 2
    max_concurrency: int = 1
    max_image_bytes: int = 64 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_path=os.environ.get("MODEL_PATH", cls.model_path),
            model_name=os.environ.get("MODEL_NAME", cls.model_name),
            model_hub_repo=os.environ.get("MODEL_HUB_REPO", cls.model_hub_repo),
            model_revision=os.environ.get("MODEL_REVISION", cls.model_revision),
            torch_threads=_int("TORCH_NUM_THREADS", cls.torch_threads),
            max_concurrency=_int("MAX_CONCURRENCY", cls.max_concurrency),
            max_image_bytes=_int("MAX_IMAGE_BYTES", cls.max_image_bytes),
        )
