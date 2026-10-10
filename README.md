# embeddinggemma-api

This repository provides a generic multimodal embedding service for Google's
EmbeddingGemma 2. It is deliberately unaware of Immich and exposes no
Immich-specific `/predict` contract. The image uses the PyTorch XPU wheels and
can run on CPU or an Intel GPU.

## Runtime choice

`MODEL_DEVICE` selects `cpu` or `xpu`; it defaults to `cpu` for local use. The
Kubernetes deployment sets it to `xpu`. When XPU is requested, startup fails
instead of silently falling back to CPU if PyTorch cannot initialize an Intel
GPU. The ready endpoint and request logs report the selected device.

Inference uses FP32 and the official Sentence Transformers integration. Audio
is disabled with `audio_config=None`; text and image encoders remain active.

The service uses the official Sentence Transformers integration for the
current cluster. Current vLLM main/nightly includes an
`EmbeddingGemma2Model` implementation and its official CPU image was tested
as a candidate, so the older claim that vLLM does not support EmbeddingGemma 2
is no longer correct. The candidate image
`vllm/vllm-openai-cpu:nightly-x86_64` exited with code 132 (SIGILL) before
x86-64-v2-era CPU flags and no AVX2/AVX512, while the official vLLM CPU build
requires AVX2 at minimum. Stock vLLM CPU is therefore not deployable on these
nodes, so this service remains the selected implementation. It loads
`google/embeddinggemma-2` at the pinned revision
`914f7f89142e33e77833254d9c9b90c3cef7303b` with:

- `sentence-transformers==6.1.0`
- `transformers==5.19.0`
- XPU `torch==2.14.0` and `torchvision==0.29.1`
- `torch.float32`; FP16 is intentionally not used
- `audio_config=None`, leaving text and image encoders active

PyTorch XPU is being evaluated on the cluster's 12th-generation Intel iGPU.
This GPU generation is not listed among the hardware validated in the current
PyTorch XPU guide, so compatibility depends on the live inference check.
The runtime image bundles the Intel Level Zero loader and compute driver
userspace libraries; the Talos worker supplies the i915 kernel driver and
firmware, while the Kubernetes device plugin passes the allocated DRM device.

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
