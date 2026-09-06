#!/usr/bin/env bash
# Leg table + launcher for the ninfer-vs-q27 head-to-head.
#
# Four legs, chosen so the quality half has a controlled pair. Bytes on disk:
#
#   q4s    15.46 GB   q27, canonical anchor f64e7c02
#   nint   17.50 GB   ninfer groupwise-int -- the FORMAT control
#   q5f    18.22 GB   q27, bpw-matched to nvfp4 within 0.5%
#   nvfp4  18.32 GB   ninfer NVFP4, their headline
#
# The 08-15 A/B compared q4s against nvfp4 at +18% bits and read the delta as
# engine quality. q5f closes that.
#
# Each leg boots in one of two configs:
#   agentic  bare serving defaults, the config real traffic would hit
#   ladder   8 slots / 16K ctx, the 08-14 concurrency protocol
#
# Every leg gets a FRESH upstream port: ninfer lacks SO_REUSEADDR, so rebinding
# a just-benched port fails on TIME-WAIT.

set -uo pipefail

Q27_BIN=/mnt/ai/projects/q27/build/q27-server
NINFER_BIN=/mnt/ai/projects/ninfer/build/apps/ninfer-serve
NINFER_MASTER_BIN=/mnt/ai/projects/ninfer-master/build/apps/ninfer-serve
Q27_TOK=/mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp.tok
Q38_TOK=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.tok
MODELS=/mnt/ai/models

LLAMA_BIN=/mnt/ai/projects/llama.cpp/build/bin/llama-server
LLAMA_GGUF=/mnt/ai/models/qwen36-27b-mtp-gguf/Qwen3.6-27B-MTP-Q5_K_M.gguf
# llama38: the Qwen3.8 competitor leg. Q5_K_M at 19.5 GB is within 0.3 GB of
# q27's q6 tier, so it is the size-matched comparison. The 3.8 template raises
# on any system message that is not first, which Claude Code sends (the skills
# listing at messages[1]); qwen38_sysinline.jinja is the stock template with
# that one branch rendering the turn in place instead of raising -- the same
# file the 2026-08-21 agentic cross-test ran on.
LLAMA38_GGUF=/mnt/ai/models/qwen38-27b-mtp-gguf/Qwen3.8-27B-MTP-Q5_K_M.gguf
LLAMA38_TEMPLATE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/qwen38_sysinline.jinja"
VLLM_MODEL=unsloth/Qwen3.6-27B-NVFP4
VLLM_IMAGE=vllm/vllm-openai:nightly
VLLM_CONTAINER=ab-vllm
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

leg_engine() {
  case "$1" in
    q4s|q5f|q38|q38q4s) echo q27 ;;
    nint|nvfp4|nvfp4m) echo ninfer ;;
    llama|llama38) echo llama ;;
    vllm) echo vllm ;;
    *) echo "?" ;;
  esac
}

leg_artifact() {
  case "$1" in
    q4s)   echo $MODELS/qwen36-27b-mtp/qwen36-27b-mtp-q4s.q27 ;;
    q5f)   echo $MODELS/qwen36-27b-mtp/qwen36-27b-mtp-q5f.q27 ;;
    q38)    echo $MODELS/qwen38-27b-mtp/qwen38-27b-mtp.q27 ;;
    q38q4s) echo $MODELS/qwen38-27b-mtp/qwen38-27b-mtp-q4s.q27 ;;
    nint)  echo $MODELS/ninfer/qwen3_6_27b.ninfer ;;
    nvfp4) echo $MODELS/ninfer/qwen3_6_27b_nvfp4.ninfer ;;
    nvfp4m) echo $MODELS/ninfer/qwen38-nvfp4-release/qwen3_8_27b_nvfp4.ninfer ;;
    llama) echo $LLAMA_GGUF ;;
    llama38) echo $LLAMA38_GGUF ;;
    vllm)  echo "$VLLM_MODEL" ;;   # HF id, resolved from the local cache
  esac
}

leg_tok() {
  case "$1" in
    q38|q38q4s) echo "$Q38_TOK" ;;
    *) echo "$Q27_TOK" ;;
  esac
}

# distinct upstream port per (leg, config) -- see TIME-WAIT note above
leg_port() {
  local base
  case "$1" in
    q4s) base=8110;; nint) base=8120;; q5f) base=8130;; nvfp4) base=8140;;
    llama) base=8150;; vllm) base=8160;;
    q38) base=8170;; q38q4s) base=8180;; llama38) base=8190;;
    nvfp4m) base=8200;;
  esac
  case "$2" in agentic) echo $((base+0));; ladder) echo $((base+1));; quality) echo $((base+2));; esac
}

# Boot a leg. $1=leg $2=config $3=logfile. Echoes the upstream port.
leg_start() {
  local leg="$1" cfg="$2" log="$3"
  local eng port art
  eng=$(leg_engine "$leg"); port=$(leg_port "$leg" "$cfg"); art=$(leg_artifact "$leg")

  # vllm's "artifact" is an HF id resolved from the local cache, not a path
  if [ "$eng" != vllm ] && [ ! -r "$art" ]; then
    echo "MISSING artifact: $art" >&2; return 1
  fi

  if [ "$eng" = q27 ]; then
    # fp8 KV + PMIN + auto-maxd are the measured Claude-Code defaults; the
    # ladder config additionally pins the 08-14 concurrency protocol.
    if [ "$cfg" = ladder ]; then
      Q27_KV=fp8 Q27_BATCH=1 Q27_PMIN=0.5 \
        nohup "$Q27_BIN" "$art" "$(leg_tok "$leg")" --port "$port" --slots 8 --ctx 16384 \
        >"$log" 2>&1 &
    else
      Q27_KV=fp8 \
        nohup "$Q27_BIN" "$art" "$(leg_tok "$leg")" --port "$port" >"$log" 2>&1 &
    fi
  elif [ "$eng" = llama ]; then
    # --device CUDA0 (+ --device-draft): left alone, llama.cpp SPLITS across
    # both GPUs -- measured 16210 MiB on the 5090 and 11370 MiB on the 3090,
    # where it also contends with the resident transcriber. Every other leg is
    # 5090-only, so the split would have made this leg incomparable (and, being
    # a 24GB Ampere card, slower per byte).
    #
    # --reasoning off: llama.cpp THINKS by default on this checkpoint and would
    # be the only leg generating reasoning tokens. It cannot be fixed at the
    # request layer here -- llama.cpp wants chat_template_kwargs.enable_thinking,
    # which is exactly the form ninfer rejects and the tap rewrites away. Server
    # side is the only place all four engines can be set to no-think.
    #
    # q8_0 KV matches the 2026-07-14 cross-engine protocol. --parallel N SPLITS
    # --ctx-size across slots (the same trap as ninfer's --kv-capacity), so the
    # ladder asks for 8 x 16384 as one total rather than per-slot.
    # --ctx-checkpoints: llama.cpp keeps recurrent-state (GDN) checkpoints PER
    # SLOT, and the default is 32. At --parallel 8 that asked for an 8379 MiB
    # "rs cache" on top of the weights and OOM'd the ladder outright. 4 is the
    # value the published 256K-on-24GB config uses for the same reason. This is
    # the third engine in this run to pay for GDN state in memory: ninfer caps
    # it at two resume offsets, q27 folds it, llama.cpp checkpoints it.
    # 6 slots, not 8: the GDN recurrent-state ("rs cache") costs ~1047 MiB PER
    # SLOT on top of 18.19 GiB of weights, and --parallel 8 asks for 8379 MiB
    # and OOMs. Measured ceiling on this card is 6 (boots at 29932 MiB).
    # q27 reaches 8 at 16K because M1 record-then-fold cut its per-slot GDN
    # cost; this is the same architectural bill paid two different ways.
    local ctx=131072 par=1 ckpt=32
    [ "$cfg" = ladder ] && { ctx=$((6*16384)); par=6; ckpt=4; }
    nohup "$LLAMA_BIN" -m "$art" --port "$port" --host 127.0.0.1 \
      --device CUDA0 --device-draft CUDA0 -ngl 999 \
      --ctx-size "$ctx" --parallel "$par" --ctx-checkpoints "$ckpt" \
      --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
      --spec-type draft-mtp --spec-draft-n-max 6 \
      --reasoning off \
      $([ "$leg" = llama38 ] && printf -- '--chat-template-file %s' "$LLAMA38_TEMPLATE") \
      --jinja --alias "$leg" >"$log" 2>&1 &
    echo $! >"${log%.log}.pid"
    echo "$port"
    return 0

  elif [ "$eng" = vllm ]; then
    # Mirrors 5090-local-llm/scripts/vllm-serve's unsloth-nvfp4-27b-mtp key,
    # MINUS its --override-generation-config temperature:0. Every other leg runs
    # its own shipped sampler defaults; forcing greedy on vLLM alone would make
    # it the only deterministic engine in the set. enable_thinking:false is KEPT
    # -- that matches the thinking normalization applied to every other leg.
    #
    # NO --speculative-config. vLLM's MTP path produces CORRUPTED generations
    # under sustained agentic traffic on this checkpoint -- incoherent token
    # soup returned as HTTP 200. Isolated 2026-08-17 over 4 instances x 3
    # variants: spec ON = 1/4 gold with 3 sessions dead at turn 1; spec OFF
    # (cache still on) = 3/4 gold, all sessions running 6-20 turns; spec OFF +
    # cache OFF = 3/4 gold. Single variable, and it is the speculative path,
    # not the prefix cache. vLLM is therefore measured in its best WORKING
    # configuration.
    #
    # --served-model-name lists the leg name AND every model id Claude Code
    # may ask for. It uses a SECOND model for background work
    # (claude-haiku-4-5-20251001); aliasing only the main one let two instances
    # run and then 404'd the rest at turn 1. vLLM's
    # Anthropic endpoint VALIDATES the model field and 404s on a mismatch, while
    # q27's and ninfer's are lenient there. Claude Code sends its own model id.
    # Aliasing server-side keeps the request bytes identical across all legs
    # rather than rewriting them in the tap for one engine.
    docker rm -f "$VLLM_CONTAINER" >/dev/null 2>&1
    # util 0.93, not the 0.96 in the old serve notes: the desktop compositor +
    # sunshine hold ~1.2 GB, so 0.96 asks for 30.08 GiB against 29.83 free and
    # the engine core refuses to start. Every other leg runs against the same
    # baseline, so this is not a handicap -- it is the real free-memory budget.
    # --gpus device=0 pins the 5090; `all` exposes the 3090 too and vLLM warns
    # about device ordering (the 3090 also has the transcriber resident).
    # 106496: vLLM VALIDATES prompt + max_tokens <= max_model_len and 400s on
    # overflow, while q27 and ninfer clamp. Claude Code asks for max_tokens
    # 64000 as a ceiling it never approaches (largest observed completion in
    # this run: 2166 tokens), so the leg needs headroom for 64000 + a ~40K
    # agentic prompt. vLLM's own estimate at util 0.93 is 107200 max.
    local maxlen=106496 util=0.93 seqs=16
    [ "$cfg" = ladder ] && { maxlen=16384; seqs=8; }
    docker run -d --name "$VLLM_CONTAINER" --gpus '"device=0"' --ipc=host \
      -p ${port}:8000 -v "${HF_CACHE}:/root/.cache/huggingface" \
      -e PYTORCH_ALLOC_CONF=expandable_segments:True \
      "$VLLM_IMAGE" \
      --model "$VLLM_MODEL" --tensor-parallel-size 1 \
      --gpu-memory-utilization "$util" --max-model-len "$maxlen" \
      --max-num-seqs "$seqs" --host 0.0.0.0 \
      --served-model-name "$leg" claude-opus-4-8 \
                          claude-haiku-4-5-20251001 claude-sonnet-4-5 \
      --enable-auto-tool-choice --tool-call-parser qwen3_coder \
      --trust-remote-code --kv-cache-dtype fp8 \
      --quantization compressed-tensors \
      --default-chat-template-kwargs '{"enable_thinking":false}' \
      >"$log" 2>&1
    echo "docker:$VLLM_CONTAINER" >"${log%.log}.pid"
    echo "$port"
    return 0

  else
    # ninfer flags mirror the 08-15 A/B leg exactly, plus --request-log-jsonl
    # for their own per-request record (kept as a cross-check on the tap, not
    # as the reported number).
    local extra=()
    # --kv-capacity is a TOTAL token budget, not per-sequence. Passing only
    # --max-context 16384 makes auto-capacity resolve the whole pool to 16384,
    # so exactly ONE 16K sequence fits and the ladder serializes -- a flat
    # aggregate that looks like "no batch scaling" but is a config artifact.
    # 8 x 16384 mirrors q27's --slots 8 --ctx 16384 on the other side.
    [ "$cfg" = ladder ] && extra+=(--max-concurrency 8 --max-context 16384 --kv-capacity 131072)
    [ "$cfg" != ladder ] && extra+=(--max-context 262144)
    # --model-id: ninfer VALIDATES the request's `model` field on
    # /v1/chat/completions and 400s with model_not_found on a mismatch (its
    # Anthropic endpoint is lenient, which is why the agentic arm worked while
    # quality scored 0/75). q27 ignores the field entirely. Naming the artifact
    # after the leg makes one client config work against both engines.
    local nbin="$NINFER_BIN" spec=(--spec mtp --draft-tokens 3 --lm-head-draft)
    # nvfp4m: ninfer MASTER + their DFlash2 drafter (their current best on 3.8)
    if [ "$leg" = nvfp4m ]; then nbin="$NINFER_MASTER_BIN"; spec=(--spec dflash2 --draft-tokens 7); fi
    nohup "$nbin" "$art" --port "$port" --model-id "$leg" \
      --kv-dtype int8 "${spec[@]}" \
      --prefill-chunk 1024 --no-thinking \
      --request-log-jsonl "${log%.log}.reqlog.jsonl" \
      "${extra[@]}" >"$log" 2>&1 &
  fi
  echo $! >"${log%.log}.pid"
  echo "$port"
}

# Block until /health answers or we give up. $1=port $2=timeout_s
leg_wait() {
  local port="$1" to="${2:-300}" i=0
  while [ $i -lt "$to" ]; do
    case "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health" 2>/dev/null)" in
      200|401) echo "up after ${i}s"; return 0 ;;
    esac
    sleep 1; i=$((i+1))
  done
  echo "TIMEOUT after ${to}s" >&2; return 1
}

leg_stop() {
  local log="$1" pidf="${1%.log}.pid"
  [ -r "$pidf" ] || return 0
  local pid; pid=$(cat "$pidf")
  if [ "${pid#docker:}" != "$pid" ]; then
    docker logs "${pid#docker:}" >"${log%.log}.container.log" 2>&1 || true
    docker rm -f "${pid#docker:}" >/dev/null 2>&1
    rm -f "$pidf"; sleep 8; return 0
  fi
  kill "$pid" 2>/dev/null
  for _ in $(seq 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -9 "$pid" 2>/dev/null
  rm -f "$pidf"
  # let VRAM actually drain before the next leg sizes itself against it
  sleep 5
}

# Peak VRAM on GPU0 right now, MiB.
vram_now() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0; }
