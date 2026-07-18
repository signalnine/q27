# q27 RunPod Serverless image -- PREBUILT binaries (use only if your worker
# pool is CONFIRMED on NVIDIA driver r580+; the v0.3.1 tarball links the
# CUDA 13.2 runtime). Otherwise use ./Dockerfile.source (builds against 12.8,
# r570+ floor). Smaller image, no compile step.
#
# Weights are NOT baked in -- attach a network volume (see README).

FROM nvidia/cuda:13.2.0-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir --break-system-packages runpod
RUN curl -fsSL https://github.com/signalnine/q27/releases/download/v0.3.1/q27-v0.3.1-linux-x86_64.tar.gz \
      | tar xz -C /opt && mv /opt/q27-v0.3.1 /opt/q27
COPY handler.py /handler.py
ENV Q27_BIN=/opt/q27/q27-server-w8 \
    Q27_MODEL=/runpod-volume/qwen36-27b-mtp-q4s.q27 \
    Q27_TOK=/runpod-volume/qwen36-27b-mtp.tok \
    Q27_KV=turbo3
CMD ["python3", "-u", "/handler.py"]
