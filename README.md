# embeddinggemma-api

This repository provides a generic multimodal embedding service for Google's
EmbeddingGemma 2. It is deliberately unaware of Immich and exposes no
Immich-specific `/predict` contract. It can run with Sentence Transformers or
proxy to a colocated llama.cpp server.

## Runtime choice

`EMBEDDING_BACKEND=llama.cpp` selects the llama.cpp proxy. It forwards to a
colocated server, preserving the `/v1/embeddings` contract and adding the
EmbeddingGemma 2 query/document prefixes before text inputs. The Kubernetes
deployment uses llama.cpp's SYCL backend and requests one Intel iGPU.

Without `EMBEDDING_BACKEND`, `MODEL_DEVICE` selects `cpu` or `xpu` for the
Sentence Transformers path and defaults to `cpu`. When XPU is requested,
startup fails rather than silently falling back to CPU. The ready endpoint and
request logs report the selected device.

The Sentence Transformers path loads `google/embeddinggemma-2` at the pinned revision
`914f7f89142e33e77833254d9c9b90c3cef7303b` with:

- `sentence-transformers==6.1.0`
- `transformers==5.19.0`
- XPU `torch==2.14.0` and `torchvision==0.29.1`
- `torch.float32`; FP16 is intentionally not used
- `audio_config=None`, leaving text and image encoders active

The Kubernetes runtime uses llama.cpp's `server-intel` image with SYCL. The
GGUF model and multimodal projector are pinned to a conversion of the same
`google/embeddinggemma-2` revision used by the Sentence Transformers path:
`914f7f89142e33e77833254d9c9b90c3cef7303b`. The converted files use the
checkpoint's native BF16 weights; the projector keeps both text and image
inputs available.

The native model output is 768 dimensions. The service validates finite values
and unit L2 norm before returning them.

## API

`POST /v1/embeddings` accepts the normal text shape:

```json
{"model":"embeddinggemma-2","input":"猫がソファにいる写真","input_type":"query"}
```

For images, the service accepts an explicit data URL content block. Remote URLs
are rejected so inference does not need network access:

```json
{
  "model": "embeddinggemma-2",
  "input": [{"content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]}],
  "input_type": "document"
}
```

`input_type=query` applies `task: search result | query: ` to text.
`input_type=document` applies `title: none | text: ` to text and no prefix to
images. `/health/live` only reports process liveness; `/health/ready` succeeds
only after the llama.cpp server is ready.

## Cache and security

Kubernetes downloads the pinned model files into an ephemeral per-pod cache
using the existing llm-gateway Hugging Face cache script. The API runtime uses
a non-root UID, a read-only root filesystem, no service account token, and
bounded concurrency. The service is exposed only as a ClusterIP and its Cilium
policy allows ingress from `litellm-proxy` and the Kubernetes host for
probes/debugging; it does not allow ingress from the `immich` namespace.
