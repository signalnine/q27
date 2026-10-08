# q27 RunPod Serverless image -- PREBUILT binaries (use only if your worker
# pool is CONFIRMED on NVIDIA driver r580+; the v0.14.4 tarball links the
# CUDA 13.2 runtime statically and needs glibc 2.38+, which this Ubuntu 24.04
# base has). Otherwise use ./Dockerfile.source (builds against 12.8, r570+
# floor). Smaller image, no compile step. The tarball also carries
# q27-server-12g (sm_86 only) for the 8-12 GB Bonsai 2 packs.
#
# Weights are NOT baked in -- attach a network volume (see README).

FROM nvidia/cuda:13.2.0-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir --break-system-packages runpod
ARG Q27_VERSION=0.14.4
ARG Q27_SHA256=1fe95631b03ebefbf9dbc4bba9cb3f9ea6734ed133626d5bae1e061bdba8f781
RUN curl -fsSL -o /tmp/q27.tgz https://github.com/signalnine/q27/releases/download/v${Q27_VERSION}/q27-v${Q27_VERSION}-linux-x86_64.tar.gz \
    && echo "${Q27_SHA256}  /tmp/q27.tgz" | sha256sum -c - \
    && tar xzf /tmp/q27.tgz -C /opt && mv /opt/q27-v${Q27_VERSION} /opt/q27 && rm /tmp/q27.tgz
COPY handler.py /handler.py
ENV Q27_BIN=/opt/q27/q27-server-w8 \
    Q27_MODEL=/runpod-volume/qwen36-27b-mtp-q4s.q27 \
    Q27_TOK=/runpod-volume/qwen36-27b-mtp.tok
# KV format: the server picks per arch (fp8 on sm_89+, turbo5k on sm_86);
# set Q27_KV=turbo3 for the deepest window.
CMD ["python3", "-u", "/handler.py"]
