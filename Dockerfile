FROM python:3.12-slim@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9 AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

WORKDIR /build
RUN python -m pip install --no-cache-dir "uv==0.12.23"
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

FROM python:3.12-slim-trixie@sha256:2b4f19dae3a777dfc3b76730bda1e82e1f66ab2a2686fa93ca78edbfb4f04ffe AS runtime

ARG INTEL_COMPUTE_RUNTIME_VERSION=26.22.38646.4
ARG INTEL_IGC_VERSION=2.36.3
ARG INTEL_IGC_BUILD=21719
ARG LEVEL_ZERO_VERSION=1.28.6

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl; \
    mkdir -p /tmp/intel-xpu-runtime; \
    curl -fSL "https://github.com/intel/compute-runtime/releases/download/${INTEL_COMPUTE_RUNTIME_VERSION}/libze-intel-gpu1_${INTEL_COMPUTE_RUNTIME_VERSION}-0_amd64.deb" -o /tmp/intel-xpu-runtime/libze-intel-gpu1.deb; \
    curl -fSL "https://github.com/intel/compute-runtime/releases/download/${INTEL_COMPUTE_RUNTIME_VERSION}/libigdgmm12_22.10.0_amd64.deb" -o /tmp/intel-xpu-runtime/libigdgmm12.deb; \
    curl -fSL "https://github.com/intel/intel-graphics-compiler/releases/download/v${INTEL_IGC_VERSION}/intel-igc-core-2_${INTEL_IGC_VERSION}+${INTEL_IGC_BUILD}_amd64.deb" -o /tmp/intel-xpu-runtime/intel-igc-core-2.deb; \
    curl -fSL "https://github.com/oneapi-src/level-zero/releases/download/v${LEVEL_ZERO_VERSION}/libze1_${LEVEL_ZERO_VERSION}%2Bu24.04_amd64.deb" -o /tmp/intel-xpu-runtime/libze1.deb; \
    echo "8bef9f24e03f826f93c076081bda13c6ac3afbd9e42b9fb8f298fab652330e2f  /tmp/intel-xpu-runtime/libze-intel-gpu1.deb" | sha256sum -c -; \
    echo "6031a63d6e8a12ce61c14efc15f2c8e727061286e3820b8594e6d00615e04d54  /tmp/intel-xpu-runtime/libigdgmm12.deb" | sha256sum -c -; \
    echo "9e0975ac75015b431ebb2da81a802b9fd1e28a3c270313a97569cd1e6a6c6048  /tmp/intel-xpu-runtime/intel-igc-core-2.deb" | sha256sum -c -; \
    echo "8af9dc06d9684a20a3f43754fe109968fba2ba085cfa572a4cc6ef0272f9bcd1  /tmp/intel-xpu-runtime/libze1.deb" | sha256sum -c -; \
    apt-get install -y --no-install-recommends /tmp/intel-xpu-runtime/*.deb; \
    apt-get purge -y --auto-remove curl; \
    rm -rf /tmp/intel-xpu-runtime /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY embeddinggemma_api /app/embeddinggemma_api

USER 10001:10001
EXPOSE 8080
ENTRYPOINT ["uvicorn", "embeddinggemma_api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--no-access-log"]
