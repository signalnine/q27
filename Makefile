CXX       ?= g++
CXXFLAGS  ?= -O2 -std=c++17 -Wall -Wextra
PYTHON    ?= python3
NVCC      ?= /usr/local/cuda/bin/nvcc
# sm_120 = RTX 5090, sm_89 = RTX 4090/Ada (needs CUDA 12.4+ for e4m3 MMA),
# sm_86 = RTX 3090 (fallback device for tests)
NVCCFLAGS ?= -O2 -std=c++17 -gencode arch=compute_86,code=sm_86 \
             -gencode arch=compute_89,code=sm_89 \
             -gencode arch=compute_120,code=sm_120 -Xcompiler -Wall

.PHONY: all clean test-inspect test-repack test-repack-canonical test-metal-backend metal-engine test-metal-contracts test-metal test-metal-canonical check-chat-extract check-responses-integration
all: build/inspect build/test_sampling build/test_kernels build/test_argmax_tie build/q27 build/q27-server build/test_tokenizer build/test_stream_split build/test_tool_drift build/test_tool_drift_corpus build/test_think_resolve build/test_openai_bridge build/test_chat_completions_integration build/test_depthctl build/test_toolconstrain
build/q27: src/engine.cu src/engine.cuh src/dflash2.cu src/dflash2.h src/kv_pool.h src/prefill_arena.h src/blocks.cu src/prefill.cu src/kernels.cu src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp \
           src/blocks.cuh src/kernels.cuh src/spec3.cuh src/prefill.cuh src/fdmma.cuh src/turbo3.cuh src/turbo5.cuh src/device_model.h src/loader.h src/cuda_common.h src/depthctl.h src/prefix_cache.h src/prefix_ram.h build/pf4.o | build
	$(NVCC) $(NVCCFLAGS) src/engine.cu src/dflash2.cu src/blocks.cu src/prefill.cu src/kernels.cu src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp build/pf4.o -o $@

build:
	mkdir -p build

build/inspect: src/inspect.cpp src/loader.cpp src/loader.h | build
	$(CXX) $(CXXFLAGS) src/inspect.cpp src/loader.cpp -o $@
build/test_loader_contracts: src/test_loader_contracts.cpp src/loader.cpp src/loader.h | build
	$(CXX) $(CXXFLAGS) src/test_loader_contracts.cpp src/loader.cpp -o $@


test-inspect: build/inspect build/test_loader_contracts
	python3 tools/test_inspect.py ./build/inspect
	./build/test_loader_contracts

test-repack: tools/repack.py tools/test_repack_split.py tools/test_repack_split_e2e.py \
             tools/test_repack_canonical_gate.py tools/requirements-repack-test.txt
	$(PYTHON) tools/test_repack_split.py
	$(PYTHON) tools/test_repack_split_e2e.py
	$(PYTHON) tools/test_repack_canonical_gate.py

# Opt-in release gate: repack a REAL source and compare the artifact MD5 to the
# published digest. The tests above prove the split plumbing on synthetic
# fixtures; only this proves the bytes we ship. Skips cleanly when unconfigured.
#   SRC_GGUF=... [SRC_MTP_GGUF=...] CANON_MD5=... make test-repack-canonical
test-repack-canonical: tools/repack_canonical_gate.sh tools/repack.py
	bash tools/repack_canonical_gate.sh

build/test_sampling: src/test_sampling.cpp src/sampling.h | build
	$(CXX) $(CXXFLAGS) src/test_sampling.cpp -o $@

build/test_tokenizer: src/test_tokenizer.cpp src/tokenizer.cpp src/tokenizer.h src/api_common.h src/drift_capture.h src/stream_split.h src/markdown_lex.h src/toolgram.h | build
	$(CXX) $(CXXFLAGS) -DQ27_TOKENIZER_TESTING src/test_tokenizer.cpp src/tokenizer.cpp -o $@

build/test_stream_split: tools/test_stream_split.cpp src/stream_split.h src/markdown_lex.h | build
	$(CXX) $(CXXFLAGS) -I src tools/test_stream_split.cpp -o $@

# CPU-only suites: no GPU, no model, no network. tool_strict() is a
# process-lifetime memo, so the drift suite runs twice -- once tolerant, once
# strict -- because the strict leg's assertions cannot share a process with the
# tolerant ones.
.PHONY: test-tools
# extract_check first: the integration harness embeds server.cu's handle()
# byte-for-byte, and a stale copy tests logic that no longer ships (it had
# drifted for ten commits before anything noticed).
test-tools: build/test_tool_drift build/test_tool_drift_corpus build/test_openai_bridge build/test_kv_bank \
            build/test_chat_completions_integration build/test_think_resolve \
            build/test_stream_split build/test_toolconstrain build/test_template_golden \
            build/test_drift_capture build/test_drift_hook
	./tools/extract_check.sh
	./build/test_tool_drift
	Q27_TOOL_STRICT=1 ./build/test_tool_drift
	./build/test_tool_drift_corpus
	./build/test_openai_bridge
	./build/test_chat_completions_integration
	./build/test_think_resolve
	./build/test_stream_split
	./build/test_toolconstrain
	./build/test_template_golden
	./build/test_drift_capture
	./build/test_drift_hook
	./build/test_kv_bank

build/test_openai_bridge: tools/test_openai_bridge.cpp src/api_common.h src/drift_capture.h src/stream_split.h src/markdown_lex.h | build
	$(CXX) $(CXXFLAGS) -I src tools/test_openai_bridge.cpp -o $@

build/test_chat_completions_integration: tools/test_chat_completions_integration.cpp src/server.cu src/api_common.h src/drift_capture.h src/toolconstrain.h src/toolgram.h src/stream_split.h src/markdown_lex.h src/tokenizer.h | build
	$(CXX) $(CXXFLAGS) -I src tools/test_chat_completions_integration.cpp -o $@

build/replay_missed_calls: tools/replay_missed_calls.cpp src/api_common.h src/drift_capture.h src/stream_split.h src/markdown_lex.h | build
	$(CXX) $(CXXFLAGS) -I src tools/replay_missed_calls.cpp -o $@
# the same turn through the STREAMING path (holdback + splitter), which is
# what a live client sees; a batch-recovered shape can still die here
build/stream_probe: tools/stream_probe.cpp src/api_common.h src/drift_capture.h src/stream_split.h src/markdown_lex.h | build
	$(CXX) $(CXXFLAGS) -I src tools/stream_probe.cpp -o $@

build/test_template_golden: tools/test_template_golden.cpp src/api_common.h src/drift_capture.h src/stream_split.h src/markdown_lex.h | build
	$(CXX) $(CXXFLAGS) -I src tools/test_template_golden.cpp -o $@

# Fuzzing the tool-call parser. The model chooses every byte this parser sees,
# so it is attack surface (docs/SECURITY.md). Two builds:
#   make fuzz       coverage-guided libFuzzer (needs clang) -- far better reach
#   make fuzz-gcc   standalone mutator under gcc+ASan, no clang needed
# Both link ASan+UBSan; a hit aborts with a stack trace.
FUZZ_CLANG ?= clang++-18
FUZZ_GCC_DIR ?= /usr/lib/gcc/x86_64-linux-gnu/13

build/fuzz_tool_parser: tools/fuzz_tool_parser.cpp src/api_common.h src/drift_capture.h src/stream_split.h src/markdown_lex.h | build
	$(FUZZ_CLANG) --gcc-install-dir=$(FUZZ_GCC_DIR) -O1 -g -std=c++17 \
	  -fsanitize=fuzzer,address,undefined -fno-omit-frame-pointer \
	  -I src tools/fuzz_tool_parser.cpp -o $@

build/fuzz_tool_parser_gcc: tools/fuzz_tool_parser.cpp tools/fuzz_tool_parser_main.cpp src/api_common.h src/drift_capture.h src/stream_split.h src/markdown_lex.h | build
	$(CXX) -O1 -g -std=c++17 -fsanitize=address,undefined -fno-omit-frame-pointer \
	  -I src tools/fuzz_tool_parser.cpp tools/fuzz_tool_parser_main.cpp -o $@

# corpus/ is scratch: libFuzzer grows it to thousands of files. Only
# tools/fuzz_seeds (the shapes the model actually emits) is tracked.
fuzz: build/fuzz_tool_parser
	mkdir -p build/fuzz_corpus && cp -n tools/fuzz_seeds/* build/fuzz_corpus/ 2>/dev/null || true
	cp -n tools/drift_corpus/seeds/* build/fuzz_corpus/ 2>/dev/null || true
	ASAN_OPTIONS=detect_leaks=0 ./build/fuzz_tool_parser build/fuzz_corpus \
	  -max_total_time=$${FUZZ_SECONDS:-300} -max_len=32768 -print_final_stats=1

fuzz-gcc: build/fuzz_tool_parser_gcc
	ASAN_OPTIONS=detect_leaks=0:abort_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
	  ./build/fuzz_tool_parser_gcc $${FUZZ_ITERS:-200000} $${FUZZ_SEED:-1}

.PHONY: fuzz fuzz-gcc

build/render_request: tools/render_request.cpp src/api_common.h src/drift_capture.h src/tokenizer.cpp src/tokenizer.h | build
	$(CXX) $(CXXFLAGS) -I src tools/render_request.cpp src/tokenizer.cpp -o $@

build/flip_regions: tools/flip_regions.cpp src/tokenizer.cpp src/tokenizer.h | build
	$(CXX) $(CXXFLAGS) -I src tools/flip_regions.cpp src/tokenizer.cpp -o $@

build/test_tool_drift: tools/test_tool_drift.cpp src/api_common.h src/drift_capture.h src/stream_split.h src/markdown_lex.h | build
	$(CXX) $(CXXFLAGS) -I src tools/test_tool_drift.cpp -o $@

build/test_tool_drift_corpus: tools/test_tool_drift_corpus.cpp src/api_common.h src/drift_capture.h src/stream_split.h src/markdown_lex.h | build
	$(CXX) $(CXXFLAGS) -I src tools/test_tool_drift_corpus.cpp -o $@

build/test_drift_capture: tools/test_drift_capture.cpp src/drift_capture.h third_party/json.hpp | build
	$(CXX) $(CXXFLAGS) -I src tools/test_drift_capture.cpp -o $@

build/test_drift_hook: tools/test_drift_hook.cpp src/api_common.h src/drift_capture.h src/stream_split.h src/markdown_lex.h | build
	$(CXX) $(CXXFLAGS) -I src tools/test_drift_hook.cpp -o $@

# Drift corpus. Serve with Q27_DRIFT_CORPUS=<file> and every dialect-bearing
# turn is appended as one redacted JSONL record (src/drift_capture.h). This
# folds that capture into tools/drift_corpus/ -- one exemplar per shape plus a
# count -- and prints the shape histogram; the seeds it writes feed `make
# fuzz`. CORPUS defaults to the same variable the server reads.
# Records are re-keyed first (build/drift_rekey) so a capture made under an
# older shape_key() folds with today's; the redacted text itself is untouched.
CORPUS ?= $(Q27_DRIFT_CORPUS)
corpus-dedup: tools/corpus_dedup.py build/drift_rekey
	@test -n "$(CORPUS)" || { echo "usage: make corpus-dedup CORPUS=/path/to/capture.jsonl (or export Q27_DRIFT_CORPUS)"; exit 2; }
	./build/drift_rekey < $(CORPUS) > build/corpus_rekeyed.jsonl
	python3 tools/corpus_dedup.py --out tools/drift_corpus build/corpus_rekeyed.jsonl

build/drift_rekey: tools/drift_rekey.cpp src/drift_capture.h third_party/json.hpp | build
	$(CXX) $(CXXFLAGS) -I src tools/drift_rekey.cpp -o $@

# Phase 2: replay every corpus shape through the parser, diff against the
# human `intended` label (tools/corpus_label.py), print the agreement number,
# and record `current` + `constrained` into the rows. Exit 1 if a
# human-confirmed label disagrees.
build/corpus_check: tools/corpus_check.cpp src/api_common.h src/drift_capture.h src/stream_split.h src/markdown_lex.h src/toolgram.h | build
	$(CXX) $(CXXFLAGS) -I src tools/corpus_check.cpp -o $@

corpus-check: build/corpus_check
	./build/corpus_check tools/drift_corpus/corpus.jsonl --write

build/test_think_resolve: tools/test_think_resolve.cpp src/api_common.h src/drift_capture.h src/stream_split.h src/markdown_lex.h | build
	$(CXX) $(CXXFLAGS) -I src tools/test_think_resolve.cpp -o $@

build/test_auth: tools/test_auth.cpp src/api_common.h src/drift_capture.h | build
	$(CXX) $(CXXFLAGS) -I src tools/test_auth.cpp -o $@

build/test_auth_integration: tools/test_auth_integration.cpp src/api_common.h src/drift_capture.h third_party/httplib.h | build
	$(CXX) $(CXXFLAGS) -I src -I third_party -pthread tools/test_auth_integration.cpp -o $@

check-chat-extract: tools/extract_check.sh src/server.cu tools/test_chat_completions_integration.cpp
	./tools/extract_check.sh

SERVER ?= build/q27-server
check-responses-integration: $(SERVER) tools/test_responses_integration.py
	python3 tools/test_responses_integration.py --server "$(SERVER)" --model "$(MODEL)" --tokenizer "$(TOKENIZER)"
build/test_depthctl: tools/test_depthctl.cpp src/depthctl.h | build
	$(CXX) $(CXXFLAGS) tools/test_depthctl.cpp -o $@

build/test_toolconstrain: tools/test_toolconstrain.cpp src/toolconstrain.h src/toolgram.h | build
	$(CXX) $(CXXFLAGS) -I src tools/test_toolconstrain.cpp -o $@

build/test_suffixdraft: tools/test_suffixdraft.cpp src/suffixdraft.h | build
	$(CXX) $(CXXFLAGS) -I src tools/test_suffixdraft.cpp -o $@

build/test_prefix_cache: tools/test_prefix_cache.cpp src/prefix_cache.h | build
	$(CXX) $(CXXFLAGS) -I src tools/test_prefix_cache.cpp -o $@

build/width_bench: tools/width_bench.cu src/kernels.cu src/spec3.cu src/vgemm.cu src/blocks.cu src/prefill.cu src/device_model.cu src/loader.cpp | build
	$(NVCC) $(NVCCFLAGS) tools/width_bench.cu src/kernels.cu src/spec3.cu src/vgemm.cu src/blocks.cu src/prefill.cu src/device_model.cu src/loader.cpp -o $@

build/mma16_bench: tools/mma16_bench.cu src/kernels.cu src/device_model.cu src/loader.cpp | build
	$(NVCC) $(NVCCFLAGS) tools/mma16_bench.cu src/kernels.cu src/device_model.cu src/loader.cpp -o $@

build/test_kernels: src/test_kernels.cu src/kernels.cu src/prefill.cu src/blocks.cu src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp src/dflash2.cu \
                    src/kernels.cuh src/prefill.cuh src/blocks.cuh src/spec3.cuh src/fdmma.cuh src/turbo3.cuh src/turbo5.cuh src/device_model.h src/loader.h src/cuda_common.h src/sampling.h src/dflash2.h | build
	$(NVCC) $(NVCCFLAGS) src/test_kernels.cu src/kernels.cu src/prefill.cu src/blocks.cu src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp src/dflash2.cu -o $@

build/test_argmax_tie: tools/test_argmax_tie.cu src/blocks.cu src/blocks.cuh | build
	$(NVCC) $(NVCCFLAGS) tools/test_argmax_tie.cu src/blocks.cu -o $@

# Tensor-manifest gate (post-review). Positive leg needs artifacts, so it is not
# in `all`; point it at every .q27 on the box before trusting the manifest:
#   make build/test_manifest && ./build/test_manifest /path/to/*.q27
build/test_manifest: tools/test_manifest.cu src/engine.cuh src/kv_pool.h src/prefill_arena.h \
                     src/conductor.h src/prefix_cache.h src/prefix_ram.h src/blocks.cu \
                     src/prefill.cu src/kernels.cu src/spec3.cu src/vgemm.cu src/device_model.cu \
                     src/loader.cpp build/pf4.o | build
	$(NVCC) $(NVCCFLAGS) -Xcompiler -pthread tools/test_manifest.cu src/blocks.cu src/prefill.cu \
	        src/kernels.cu src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp \
	        build/pf4.o -o $@


build/q27-server: src/server.cu src/engine.cuh src/dflash2.cu src/dflash2.h src/metrics.h src/kv_pool.h src/prefill_arena.h src/conductor.h src/blocks.cu src/prefill.cu src/kernels.cu src/spec3.cu src/vgemm.cu \
                  src/device_model.cu src/loader.cpp src/tokenizer.cpp src/api_common.h src/drift_capture.h src/stream_split.h src/markdown_lex.h \
                  src/blocks.cuh src/kernels.cuh src/spec3.cuh src/prefill.cuh src/fdmma.cuh src/turbo3.cuh src/turbo5.cuh src/cuda_common.h src/toolgram.h \
                  src/depthctl.h src/toolconstrain.h src/tokenizer.h src/prefix_cache.h src/prefix_ram.h third_party/httplib.h build/pf4.o | build
	$(NVCC) $(NVCCFLAGS) -Xcompiler -pthread src/server.cu src/dflash2.cu src/blocks.cu src/prefill.cu src/kernels.cu \
	        src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp src/tokenizer.cpp build/pf4.o -o $@

clean:
	rm -rf build

build/gdn_chunk_bench: tools/gdn_chunk_bench.cu | build
	$(NVCC) $(NVCCFLAGS) tools/gdn_chunk_bench.cu -o $@

build/attn_fdw_bench: tools/attn_fdw_bench.cu | build
	$(NVCC) $(NVCCFLAGS) tools/attn_fdw_bench.cu -o $@

# sm_120a-ONLY target. The block-scaled fp4 MMA (mma.sync...kind::mxf4nvf4)
# exists only behind the arch-SPECIFIC target: with the plain
# compute_120/sm_120 gencode above, ptxas rejects the instruction and the
# failure is indistinguishable from a hardware limitation -- that exact
# omission produced the 2026-08 fp4 NO-GO verdict (falsified 2026-08-14, see
# docs/plans/2026-08-15-ninfer-steals.md). The tri-arch serving binary stays
# on NVCCFLAGS until a real kernel needs 120a; only this tool gets the flag.
MXF4FLAGS = -O2 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xcompiler -Wall
# fp4 prefill kernels (Q27_PREFILL=fp4): the ONE engine TU that needs the
# arch-specific target. Compiled as a standalone object with MXF4FLAGS and
# linked into every engine binary; its kernels have no SASS for sm_86/89 and
# are never launched there (q27k::pf4_on() gates on sm_120 at runtime).
build/pf4.o: src/pf4.cu src/pf4.h | build
	$(NVCC) $(MXF4FLAGS) -c src/pf4.cu -o $@

# src/vgemm.cu is in the link line for the T2 decode leg: the union GEMM the
# C-sweep routes batched decode to at k >= 3 is the BASELINE at decode shapes
# (gemm_q4_T is prefill-only and is never reached from a fused round).
build/microbench_mxf4: tools/microbench_mxf4.cu src/prefill.cu src/kernels.cu src/vgemm.cu src/device_model.cu src/loader.cpp \
                       src/prefill.cuh src/kernels.cuh src/vgemm.cuh src/device_model.h src/loader.h src/cuda_common.h | build
	$(NVCC) $(MXF4FLAGS) tools/microbench_mxf4.cu src/prefill.cu src/kernels.cu src/vgemm.cu src/device_model.cu src/loader.cpp -o $@

# Vendor-ceiling probe for the prefill GEMM shapes (cuBLASLt int8/fp8/fp16);
# docs/perf-attribution-prefill-2026-09-08.md section 4. Run: build/cublaslt_peak 0
build/cublaslt_peak: tools/cublaslt_peak.cu | build
	$(NVCC) -O2 -arch=sm_120a tools/cublaslt_peak.cu -lcublasLt -lcublas -o $@

# W4A8 prefill GEMM spike (prefill plan phase 2, BUILDLOG 2026-09-08 (h)):
# bitwise gate vs gemm_q4_T on the projection shapes + timing of the variants.
# Run: build/gemm_w4a8_spike [--time] [--shape ffn_gate] [--only VARIANT] [--notest]
build/gemm_w4a8_spike: tools/gemm_w4a8_spike.cu src/prefill.cu src/kernels.cu src/prefill.cuh src/kernels.cuh | build
	$(NVCC) $(MXF4FLAGS) -lineinfo -I src tools/gemm_w4a8_spike.cu src/prefill.cu src/kernels.cu -o $@

VGEMM_SRC = src/vgemm.cu src/kernels.cu src/spec3.cu src/blocks.cu src/prefill.cu \
            src/device_model.cu src/loader.cpp

# P1 gates for the flat-in-W verify weight path (docs/plans/2026-07-13-gemm-verify.md):
#   vgemm_test -- gate 3 (numerics vs the gemv on all lanes/widths + determinism)
#                 and gate 4 (regs/spill/CTA-per-SM; FAILS LOUD -- zero slack).
#   vgemm_race -- gate 6's racecheck leg. racecheck instruments every shared-memory
#                 access and cannot finish on a real 47MB weight, so this drives the
#                 identical reduce path on a synthetic shape with z > 1.
build/vgemm_test: tools/vgemm_test.cu src/vgemm.cuh $(VGEMM_SRC) | build
	$(NVCC) $(NVCCFLAGS) tools/vgemm_test.cu $(VGEMM_SRC) -o $@

build/vgemm_race: tools/vgemm_race.cu src/vgemm.cuh $(VGEMM_SRC) | build
	$(NVCC) $(NVCCFLAGS) tools/vgemm_race.cu $(VGEMM_SRC) -o $@

# E6 attribution harness: splits the weight sweep's missing bandwidth into launch
# gaps / reduce epilogue / grid tail / the tile itself, against a MEASURED SOL.
build/vgemm_e6: tools/vgemm_e6.cu src/vgemm.cuh $(VGEMM_SRC) | build
	$(NVCC) $(NVCCFLAGS) tools/vgemm_e6.cu $(VGEMM_SRC) -o $@

build/round_weight_cost: tools/round_weight_cost.cu src/vgemm.cuh $(VGEMM_SRC) | build
	$(NVCC) $(NVCCFLAGS) tools/round_weight_cost.cu $(VGEMM_SRC) -o $@

build/fdmma_test: tools/fdmma_test.cu src/fdmma.cuh | build
	$(NVCC) $(NVCCFLAGS) tools/fdmma_test.cu -o $@

build/turbo3_test: tools/turbo3_test.cu src/turbo3.cuh | build
	$(NVCC) $(NVCCFLAGS) tools/turbo3_test.cu -o $@

# turbo5 (5-bit K) format gate, docs/plans/2026-08-01-5bit-k.md phase P0.
# Depends on turbo3.cuh too: turbo5.cuh includes it for the shared WHT.
build/turbo5_test: tools/turbo5_test.cu src/turbo5.cuh src/turbo3.cuh | build
	$(NVCC) $(NVCCFLAGS) tools/turbo5_test.cu -o $@

build/i8g64_test: tools/i8g64_test.cu src/i8g64.cuh | build
	$(NVCC) $(NVCCFLAGS) tools/i8g64_test.cu -o $@

# 24GB-card (3090-class) server: Q27_W_MAX=8 drops the per-width verify
# graphs above W8 and a few record-arena rows. M1b collapsed the per-perm
# zoo for every build, so the w8 delta is now small -- kept for the widest
# fits (the historical role-set + 12x-zoo savings are engine-wide now).
# Same sources, own binary.
build/q27-server-w8: src/server.cu src/engine.cuh src/dflash2.cu src/metrics.h src/kv_pool.h src/prefill_arena.h src/conductor.h src/blocks.cu src/prefill.cu src/kernels.cu src/spec3.cu src/vgemm.cu \
                     src/device_model.cu src/loader.cpp src/tokenizer.cpp src/api_common.h src/drift_capture.h src/stream_split.h src/markdown_lex.h \
                     src/blocks.cuh src/kernels.cuh src/spec3.cuh src/prefill.cuh src/fdmma.cuh src/turbo3.cuh src/turbo5.cuh src/cuda_common.h src/toolgram.h \
                     src/depthctl.h src/toolconstrain.h src/tokenizer.h src/prefix_cache.h src/prefix_ram.h third_party/httplib.h build/pf4.o | build
	$(NVCC) $(NVCCFLAGS) -DQ27_W_MAX=8 -Xcompiler -pthread src/server.cu src/dflash2.cu src/blocks.cu src/prefill.cu src/kernels.cu \
	        src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp src/tokenizer.cpp build/pf4.o -o $@

# Continuous-batching gates (docs/plans/2026-07-14-continuous-batching.md):
#   ninv_test      -- N-invariance: per-lane weight-kernel output must be bitwise
#                     independent of union width and slot (the batching contract).
#   test_conductor -- CPU: trim policy + ConductorCore membership/round-boundary.
#   fused_smoke    -- 2-engine fused round vs solo byte-identity + conductor +
#                     A2 error-injection legs (needs the GPU + model).
build/ninv_test: tools/ninv_test.cu src/vgemm.cuh src/kernels.cuh src/blocks.cuh $(VGEMM_SRC) | build
	$(NVCC) $(NVCCFLAGS) tools/ninv_test.cu $(VGEMM_SRC) -o $@

build/test_conductor: tools/test_conductor.cpp src/conductor.h | build
	$(CXX) $(CXXFLAGS) -I src tools/test_conductor.cpp -o $@

# incremental KV entitlements (issue #42 step 2): banker's safety check, CPU
build/test_kv_bank: tools/test_kv_bank.cpp src/kv_bank.h | build
	$(CXX) $(CXXFLAGS) -I src tools/test_kv_bank.cpp -o $@

build/fused_smoke: tools/fused_smoke.cu src/engine.cuh src/kv_pool.h src/prefill_arena.h src/conductor.h src/prefix_cache.h src/prefix_ram.h src/dflash2.cu src/dflash2.h src/blocks.cu src/prefill.cu \
                   src/kernels.cu src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp build/pf4.o | build
	$(NVCC) $(NVCCFLAGS) tools/fused_smoke.cu src/dflash2.cu src/blocks.cu src/prefill.cu src/kernels.cu \
	        src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp build/pf4.o -o $@

# w16 serving build (batch mode's natural target; was hand-built since part 10)
build/q27-server-w16: src/server.cu src/engine.cuh src/metrics.h src/kv_pool.h src/prefill_arena.h src/conductor.h src/blocks.cu src/prefill.cu src/kernels.cu src/spec3.cu src/vgemm.cu \
                      src/device_model.cu src/loader.cpp src/tokenizer.cpp src/api_common.h src/drift_capture.h src/stream_split.h src/markdown_lex.h \
                      src/blocks.cuh src/kernels.cuh src/spec3.cuh src/prefill.cuh src/fdmma.cuh src/turbo3.cuh src/turbo5.cuh src/cuda_common.h src/toolgram.h \
                      src/depthctl.h src/toolconstrain.h src/tokenizer.h src/prefix_cache.h src/prefix_ram.h src/kv_pool.h src/prefill_arena.h third_party/httplib.h build/pf4.o | build
	$(NVCC) $(NVCCFLAGS) -DQ27_W_MAX=16 -Xcompiler -pthread src/server.cu src/blocks.cu src/prefill.cu src/kernels.cu \
	        src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp src/tokenizer.cpp build/pf4.o -o $@

# Native Metal backend primitives. Kept separate from the engine/CLI targets so
# this dependency cut stays buildable and testable on its own.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
METALFLAGS := $(CXXFLAGS) -Werror -fobjc-arc -pthread -I src/metal
METALLIBS := -framework Foundation -framework Metal

build/test-metal-backend: src/metal/test_metal.cpp src/metal/metal_backend.mm \
                          src/metal/metal_backend.h src/metal/q27_kernels.metal \
                          src/backend.h src/loader.cpp src/loader.h | build
	$(CXX) $(METALFLAGS) src/metal/test_metal.cpp src/metal/metal_backend.mm \
	       src/loader.cpp $(METALLIBS) -o $@

build/test-metal-ops: src/metal/test_metal_ops.cpp src/metal/metal_backend.mm \
                      src/metal/metal_backend.h src/metal/q27_kernels.metal \
                      src/backend.h src/loader.cpp src/loader.h | build
	$(CXX) $(METALFLAGS) src/metal/test_metal_ops.cpp src/metal/metal_backend.mm \
	       src/loader.cpp $(METALLIBS) -o $@

build/metal-engine.o: src/metal/metal_engine.cpp src/metal/metal_engine.h \
                      src/metal/metal_backend.h src/backend.h src/loader.h \
                      src/sampling.h third_party/json.hpp | build
	$(CXX) $(METALFLAGS) -c src/metal/metal_engine.cpp -o $@

build/test_metal_engine_contracts: src/metal/test_metal_engine_contracts.cpp \
                                   src/metal/metal_engine.cpp src/metal/metal_engine.h \
                                   src/metal/metal_backend.mm src/metal/metal_backend.h \
                                   src/metal/q27_kernels.metal src/backend.h \
                                   src/loader.cpp src/loader.h src/sampling.h \
                                   third_party/json.hpp | build
	$(CXX) $(METALFLAGS) src/metal/test_metal_engine_contracts.cpp \
	       src/metal/metal_engine.cpp src/metal/metal_backend.mm src/loader.cpp \
	       $(METALLIBS) -o $@

test-metal-contracts: build/test_metal_engine_contracts build/test_metal_model_contracts build/test_metal_stream
	@test -n "$(MODEL)" || { echo "set MODEL=...q4s.q27" >&2; exit 2; }
	@if test -n "$(BONSAI_MODEL)"; then \
		./build/test_metal_engine_contracts "$(MODEL)" "$(BONSAI_MODEL)"; \
	else \
		./build/test_metal_engine_contracts "$(MODEL)"; \
	fi
	./build/test_metal_model_contracts "$(MODEL)"
	./build/test_metal_stream

metal-engine: build/metal-engine.o

build/q27-metal: src/metal/metal_cli.cpp src/metal/metal_engine.cpp \
                 src/metal/metal_backend.mm src/metal/metal_engine.h \
                 src/metal/metal_backend.h src/metal/q27_kernels.metal \
                 src/backend.h src/loader.cpp src/loader.h src/tokenizer.cpp \
                 src/tokenizer.h src/sampling.h third_party/json.hpp | build
	$(CXX) $(METALFLAGS) src/metal/metal_cli.cpp src/metal/metal_engine.cpp \
	       src/metal/metal_backend.mm src/loader.cpp src/tokenizer.cpp \
	       $(METALLIBS) -o $@

build/test_metal_model_contracts: src/metal/test_metal_model_contracts.cpp \
                                  src/metal/metal_engine.cpp src/metal/metal_backend.mm \
                                  src/metal/metal_engine.h src/metal/metal_backend.h \
                                  src/metal/q27_kernels.metal src/backend.h src/loader.cpp | build
	$(CXX) $(METALFLAGS) src/metal/test_metal_model_contracts.cpp \
	       src/metal/metal_engine.cpp src/metal/metal_backend.mm src/loader.cpp \
	       $(METALLIBS) -o $@

test-metal-validation: build/inspect build/q27-metal
	@test -n "$(TOKENIZER)" || { echo "set TOKENIZER=...tok" >&2; exit 2; }
	python3 tools/test_inspect.py ./build/inspect ./build/q27-metal "$(TOKENIZER)"

build/q27-metal-server: src/metal/metal_server.cpp src/metal/metal_engine.cpp \
                        src/metal/metal_backend.mm src/metal/metal_engine.h \
                        src/metal/metal_backend.h src/metal/stream_format.h \
                        src/metal/serving_policy.h src/metal/disk_snapshot_store.h \
                        src/metal/snapshot_evict.h \
                        src/metal/q27_kernels.metal src/api_common.h src/drift_capture.h src/stream_split.h \
                        src/toolconstrain.h src/toolgram.h src/backend.h src/loader.h src/loader.cpp \
                        src/sampling.h src/tokenizer.h src/tokenizer.cpp \
                        third_party/httplib.h third_party/json.hpp | build
	$(CXX) $(METALFLAGS) -I src/metal src/metal/metal_server.cpp src/metal/metal_engine.cpp \
	        src/metal/metal_backend.mm src/loader.cpp src/tokenizer.cpp $(METALLIBS) -o $@

build/q27-metal-server-test: src/metal/metal_server.cpp src/metal/metal_engine.cpp \
                             src/metal/metal_backend.mm src/metal/metal_engine.h \
                             src/metal/metal_backend.h src/metal/stream_format.h \
                             src/metal/serving_policy.h src/metal/disk_snapshot_store.h \
                             src/metal/snapshot_evict.h \
                             src/metal/q27_kernels.metal src/api_common.h src/drift_capture.h src/stream_split.h \
                             src/toolconstrain.h src/toolgram.h src/backend.h src/loader.h src/loader.cpp \
                             src/sampling.h src/tokenizer.h src/tokenizer.cpp \
                             third_party/httplib.h third_party/json.hpp | build
	$(CXX) $(METALFLAGS) -DQ27_METAL_TEST_FAILPOINTS=1 -I src/metal \
	        src/metal/metal_server.cpp src/metal/metal_engine.cpp src/metal/metal_backend.mm \
	        src/loader.cpp src/tokenizer.cpp $(METALLIBS) -o $@

build/test_metal_stream: src/metal/test_metal_stream.cpp src/metal/stream_format.h \
                         src/metal/serving_policy.h src/api_common.h src/drift_capture.h src/stream_split.h \
                         third_party/json.hpp | build
	$(CXX) $(CXXFLAGS) -I src/metal src/metal/test_metal_stream.cpp -o $@

build/test_snapshot_store_shared: tools/test_snapshot_store_shared.cpp \
                                  src/metal/disk_snapshot_store.h src/metal/snapshot_evict.h | build
	$(CXX) $(CXXFLAGS) -I src/metal tools/test_snapshot_store_shared.cpp -o $@

test-metal-recovery: build/q27-metal-server-test src/metal/test_server_recovery.py
	@test -n "$(MODEL)" || { echo "set MODEL=...q4s.q27" >&2; exit 2; }
	@test -n "$(TOKENIZER)" || { echo "set TOKENIZER=...tok" >&2; exit 2; }
	python3 src/metal/test_server_recovery.py ./build/q27-metal-server-test "$(MODEL)" "$(TOKENIZER)"

test-metal: test-metal-backend build/test_metal_stream build/test_snapshot_store_shared build/q27-metal
	./build/test_metal_stream
	./build/test_snapshot_store_shared
	@test -n "$(MODEL)" || { echo "set MODEL=...q4s.q27" >&2; exit 2; }
	@test -n "$(TOKENIZER)" || { echo "set TOKENIZER=...tok" >&2; exit 2; }
	@if ./build/q27-metal "$(MODEL)" "$(TOKENIZER)" --tokens 760 --prompt "" >/dev/null 2>&1; then \
		echo "Metal CLI accepted --tokens with an empty --prompt" >&2; exit 1; fi
	@if ./build/q27-metal "$(MODEL)" "$(TOKENIZER)" --tokens "" --prompt test >/dev/null 2>&1; then \
		echo "Metal CLI accepted an empty --tokens with --prompt" >&2; exit 1; fi
	./build/q27-metal "$(MODEL)" "$(TOKENIZER)" --validate-only
	./build/q27-metal "$(MODEL)" "$(TOKENIZER)" --tokens 760,6511,314,9338,369 \
	       -n 2 --ctx 16 --mtp 4 --dump-token-ids build/metal-smoke.ids

test-metal-canonical: build/q27-metal
	@test -n "$(MODEL)" || { echo "set MODEL=...q4s.q27" >&2; exit 2; }
	@test -n "$(TOKENIZER)" || { echo "set TOKENIZER=...tok" >&2; exit 2; }
	tools/metal_canonical_gate.sh "$(MODEL)" "$(TOKENIZER)"

test-metal-backend: build/test-metal-backend build/test-metal-ops
	./build/test-metal-backend
	./build/test-metal-ops
else
test-metal-backend:
	@echo "test-metal-backend requires macOS" >&2; exit 1
metal-engine:
	@echo "metal-engine requires macOS" >&2; exit 1
test-metal-contracts:
	@echo "test-metal-contracts requires macOS" >&2; exit 1
test-metal-recovery:
	@echo "test-metal-recovery requires macOS" >&2; exit 1
test-metal-validation:
	@echo "test-metal-validation requires macOS" >&2; exit 1

test-metal:
	@echo "test-metal requires macOS" >&2; exit 1

test-metal-canonical:
	@echo "test-metal-canonical requires macOS" >&2; exit 1
endif

build/dflash2_smoke: tools/dflash2_smoke.cu src/dflash2.cu src/kernels.cu src/spec3.cu src/vgemm.cu src/blocks.cu src/prefill.cu src/device_model.cu src/loader.cpp | build
	$(NVCC) $(NVCCFLAGS) tools/dflash2_smoke.cu src/dflash2.cu src/kernels.cu src/spec3.cu src/vgemm.cu src/blocks.cu src/prefill.cu src/device_model.cu src/loader.cpp -o $@
