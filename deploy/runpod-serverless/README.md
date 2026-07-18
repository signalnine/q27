# q27 on RunPod Serverless (stateless single-shot)

Runs q27 as a RunPod Serverless worker for **stateless** inference:
single-shot text completions and one-turn chat. Each job is independent;
no conversation state carries between jobs.

> **Not for multi-turn Claude Code.** q27's speed edge on agentic traffic
> comes from checkpointing GDN recurrent state so warm turns skip
> re-prefill. Serverless scatters a conversation's turns across workers,
> which destroys that cache. For CC serving, run a persistent pod with
> session affinity instead. This deployment is for the stateless case,
> where q27's decode speed is pure upside.

## How it works

The handler boots `q27-server` **once per worker** at module load, not per
request -- q27 preallocates weights + the whole CUDA-graph zoo at boot, so
per-request boots would pay that every time. Each job forwards to the local
server. RunPod FlashBoot snapshots the ready worker, so warm starts are
fast; a truly-cold worker pays the weight load + graph build once.

## 1. Weights on a network volume (do NOT bake into the image)

The `.q27` weights are 15-28 GB -- put them on a RunPod **network volume**,
not the container image. Create a volume, then on any pod/one-off:

```bash
cd /runpod-volume
huggingface-cli download signalnine/Qwen3.6-27B-MTP-q27 \
  --include qwen36-27b-mtp-q4s.q27 qwen36-27b-mtp.tok CHECKSUMS.md5 \
  --local-dir . --local-dir-use-symlinks False
md5sum -c CHECKSUMS.md5 --ignore-missing
```

The volume mounts at `/runpod-volume` in the worker; the env defaults point
there. (q4s = 15.46 GB, the serverless sweet spot: fits 24 GB with room,
fastest decode, best context.)

## 2. Build the image

```bash
# recommended: build from source, r570+ driver floor (safe on any RunPod host)
docker build -f Dockerfile.source -t <you>/q27-serverless:v0.3.1 .

# OR, only if your worker pool is confirmed r580+ (CUDA 13 host driver):
docker build -f Dockerfile -t <you>/q27-serverless:v0.3.1 .
docker push <you>/q27-serverless:v0.3.1
```

## 3. Create the endpoint

- **GPU**: a 24-48 GB card the fatbin covers -- A5000 / A6000 (sm_86),
  4090 / L40 / L40S (sm_89). (A100 sm_80 and H100 sm_90 are NOT in the
  build and you don't need them for a 27B model.)
- Attach the network volume from step 1.
- **Container disk** can be small (weights are on the volume).
- **Cold start**: first boot on a fresh worker is minutes (weight load +
  graph zoo). Set **min workers >= 1** if you need low first-request
  latency, or accept the cold penalty and let FlashBoot keep warm workers
  ready. `Q27_SAMPLED=0` (below) shortens the boot if you serve greedy-only.

### Env vars

| var | default | note |
|---|---|---|
| `Q27_MODEL` | `/runpod-volume/qwen36-27b-mtp-q4s.q27` | tier on the volume |
| `Q27_TOK` | `/runpod-volume/qwen36-27b-mtp.tok` | tokenizer |
| `Q27_BIN` | `/opt/q27/q27-server-w8` | use `q27-server` (W12) on >24 GB cards |
| `Q27_KV` | `turbo3` | `fp8` = faster/less ctx; turbo3 = full 262K on 24 GB |
| `Q27_SAMPLED` | (unset) | `0` = greedy-only, skips ~600 MB sampled graphs + shortens boot (temperature>0 requests then 400) |
| `Q27_BOOT_TIMEOUT_S` | `600` | health-wait budget for cold boot |

## 4. Call it

```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "def fib(n):\n", "max_tokens": 128, "temperature": 0}}'
```

Chat shape (one turn):

```json
{"input": {"messages": [{"role": "user", "content": "say hi"}], "max_tokens": 32}}
```

Response: `{"text": "...", "finish_reason": "...", "usage": {...}}`.

## Throughput tip: let q27 batch concurrent jobs

q27's continuous batching is on by default. If you set the endpoint's
**per-worker concurrency** to 2 and boot with `Q27_BIN=/opt/q27/q27-server`
+ `--slots 2` (edit the handler's cmd), two concurrent jobs fuse through one
weight sweep at ~1.4x aggregate. Only worth it if your traffic is bursty
enough to co-arrive; single-concurrency is simpler and loses nothing on
steady one-at-a-time load.
