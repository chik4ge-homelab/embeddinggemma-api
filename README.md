# embeddinggemma-api

This repository provides a generic, CPU-only embedding service for Google's
EmbeddingGemma 2. It is deliberately unaware of Immich and exposes no
Immich-specific `/predict` contract.

## Runtime choice

The service uses the official Sentence Transformers integration because the
model is a unified text/image embedding model and vLLM's supported pooling
model list does not include EmbeddingGemma 2. It loads
`google/embeddinggemma-2` at the pinned revision
`914f7f89142e33e77833254d9c9b90c3cef7303b` with:

- `sentence-transformers==6.1.0`
- `transformers==5.19.0`
- CPU `torch==2.14.0` and `torchvision==0.29.1`
- `torch.float32` on CPU; FP16 is intentionally not used
- `audio_config=None`, leaving text and image encoders active

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

`input_type=query` applies the model's `SearchQuery` instruction to text.
`input_type=document` applies its `Document` instruction to text and no prefix
to images. `/health/live` only reports process liveness; `/health/ready` is
successful only after the model has loaded.

## Cache and security

Kubernetes provisions the pinned model files into a dedicated PVC using the
existing llm-gateway Hugging Face cache script. The runtime uses
`HF_HUB_OFFLINE=1`, a non-root UID, a read-only root filesystem, no service
account token, and bounded single-request concurrency. The service is exposed
only as a ClusterIP and its Cilium policy allows ingress from `litellm-proxy`
and the Kubernetes host for probes/debugging; it does not allow ingress from
the `immich` namespace.
