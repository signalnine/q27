#!/usr/bin/env bash
# Bonsai 2 27B on an 8 GB (or 12 GB) NVIDIA card with the q27 engine.
#
# What this does, in order: checks the toolchain, clones q27 and builds the
# small-card server, downloads the 6 GB pack + tokenizer from Hugging Face
# and verifies the md5s, writes a run.sh, then boots the server once and
# asks it a question so you know it works before you wire anything to it.
#
# Needs: an NVIDIA driver (r570+ for CUDA 12.8, r580+ for CUDA 13), the CUDA
# toolkit 12.8 or newer at /usr/local/cuda, gcc/g++, make, git, curl. Nothing
# here runs sudo; install those first if the preflight complains.
#
#   bash install-bonsai2-8gb.sh            # plain pack: ~37K context headless on 8 GB (3060 Ti measured)
#   bash install-bonsai2-8gb.sh --mtp      # + the MTP head: faster, 12K context on 8 GB
#   BONSAI2_DIR=/somewhere bash install-bonsai2-8gb.sh
#
# 8 GB cards: run it headless (`sudo init 3`, nothing else on the GPU): 36.9K
# context measured on a 3060 Ti. With a desktop on the card expect ~24K with the
# plain pack, and the MTP pack may not fit.
# 12 GB cards: same script, more context.
set -euo pipefail

DIR=${BONSAI2_DIR:-$HOME/bonsai2}
Q27_REF=${Q27_REF:-v0.14.1}          # the q27 release this was tested with (T3 pack + the faster T3 GEMV)
HF=https://huggingface.co/signalnine/Bonsai-2-27B-q27/resolve/main
PACK=bonsai2-27b-t3-slim.q27; STACK=0.6
if [ "${1:-}" = "--mtp" ]; then PACK=bonsai2-27b-t3-mtp-slim.q27; STACK=0.8; fi
PORT=${PORT:-8090}

say() { printf '\n== %s\n' "$*"; }
die() { printf '\n!! %s\n' "$*" >&2; exit 1; }

say "preflight"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found -- install the NVIDIA driver first"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1
NVCC=/usr/local/cuda/bin/nvcc
[ -x $NVCC ] || die "no nvcc at /usr/local/cuda/bin -- install the CUDA toolkit (12.8 or newer) from developer.nvidia.com/cuda-downloads"
CUDA_VER=$($NVCC --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1.\2/p')
CUDA_MAJ=${CUDA_VER%%.*}; CUDA_MIN=${CUDA_VER##*.}
if [ "$CUDA_MAJ" -lt 12 ] || { [ "$CUDA_MAJ" -eq 12 ] && [ "$CUDA_MIN" -lt 8 ]; }; then
  die "CUDA $CUDA_VER is too old -- the build links an sm_120a object, 12.8 is the floor"
fi
echo "CUDA toolkit $CUDA_VER"
for t in gcc g++ make git curl md5sum python3; do
  command -v $t >/dev/null || die "$t not found -- on Ubuntu: sudo apt install build-essential git curl python3"
done

say "q27 at $DIR/q27 (ref $Q27_REF)"
mkdir -p "$DIR/models"
if [ -d "$DIR/q27/.git" ]; then
  git -C "$DIR/q27" fetch -q origin
else
  git clone -q https://github.com/signalnine/q27.git "$DIR/q27"
fi
git -C "$DIR/q27" checkout -q "$Q27_REF"

say "building the small-card server (one sm_86 image; ~5 min)"
make -C "$DIR/q27" build/q27-server-12g >"$DIR/build.log" 2>&1 || { tail -30 "$DIR/build.log"; die "build failed -- see $DIR/build.log"; }
ls -la "$DIR/q27/build/q27-server-12g" | awk '{print $5" bytes", $NF}'

say "downloading the pack and tokenizer (6 GB, resumable; already-verified files are skipped)"
curl -sL --fail --retry 5 -o "$DIR/models/CHECKSUMS.md5" "$HF/CHECKSUMS.md5"
for f in qwen38-27b-mtp.tok "$PACK"; do
  if [ -s "$DIR/models/$f" ] && ( cd "$DIR/models" && grep " $f\$" CHECKSUMS.md5 | md5sum -c --quiet - ) 2>/dev/null; then
    echo "   $f: present, md5 ok"; continue
  fi
  curl -L --fail --retry 5 --retry-delay 10 -C - -o "$DIR/models/$f" "$HF/$f" || die "download of $f failed -- rerun to resume"
  ( cd "$DIR/models" && grep " $f\$" CHECKSUMS.md5 | md5sum -c - ) || die "md5 mismatch on $f -- delete it under $DIR/models and rerun"
done

say "writing $DIR/run.sh"
cat > "$DIR/run.sh" <<EOF
#!/usr/bin/env bash
# Serves Bonsai 2 on 127.0.0.1:$PORT. Extra args pass through to the server
# (e.g. --ctx 32768 to pin the window, --host 0.0.0.0 to expose it).
# Q27_FIXED_STACK_GB is the engine's measured non-KV footprint for this build;
# the KV cache takes whatever VRAM is left (turbo5k format on Ampere).
# Q27_BATCH=0 = no multi-slot conductor: one slot decodes ~20% faster without it.
cd "$DIR"
exec env Q27_FIXED_STACK_GB=$STACK Q27_BATCH=0 ./q27/build/q27-server-12g models/$PACK models/qwen38-27b-mtp.tok \\
  --slots 1 --host 127.0.0.1 --port $PORT \\
  --think --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --think-budget 0 "\$@"
EOF
chmod +x "$DIR/run.sh"

say "smoke test: booting the server once (first boot ~1-2 min: the pack is relaid for the GPU, then graphs are built)"
"$DIR/run.sh" >"$DIR/server.log" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
  if curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q 200; then break; fi
  kill -0 $SPID 2>/dev/null || { tail -20 "$DIR/server.log"; die "server died during boot -- see $DIR/server.log"; }
  sleep 2
done
grep -E "vram: free|\[pool\] (paged|ctx)|KV cache|wsum" "$DIR/server.log" | sed 's/^/   /'
BODY=$(curl -s -m 300 "http://127.0.0.1:$PORT/v1/messages" \
  -H 'content-type: application/json' -H 'x-api-key: local' -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"q27","max_tokens":120,"temperature":0,"messages":[{"role":"user","content":"In two sentences, what does a hash table do?"}]}') || true
python3 - "$BODY" <<'PY' || { echo "   raw response: ${BODY:0:300}"; tail -5 "$DIR/server.log"; die "the request failed -- see $DIR/server.log"; }
import json, sys
d = json.loads(sys.argv[1])
print("   reply:", "".join(b.get("text", "") for b in d["content"]).strip()[:400])
print("   output tokens:", d["usage"]["output_tokens"])
PY
grep -oE "tps=[0-9.]+" "$DIR/server.log" | tail -1 | sed 's/tps=/   decode t\/s: /'
kill $SPID; wait $SPID 2>/dev/null || true
trap - EXIT

say "done"
cat <<EOF
   start it:      $DIR/run.sh
   health:        curl http://127.0.0.1:$PORT/health
   Anthropic API: http://127.0.0.1:$PORT/v1/messages   (any x-api-key)
   OpenAI API:    http://127.0.0.1:$PORT/v1/chat/completions
   Claude Code:   ANTHROPIC_BASE_URL=http://127.0.0.1:$PORT ANTHROPIC_API_KEY=local claude
                  (its stock prompt + tool schemas need more than a 24K window: if the
                  [pool] ctx below is 24K or less -- a desktop on the card, or the MTP
                  pack -- run it as: claude --bare --system-prompt ".")
   logs:          $DIR/server.log (the [pool] lines say how much context you got)
EOF
