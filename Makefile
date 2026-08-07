CXX       ?= g++
CXXFLAGS  ?= -O2 -std=c++17 -Wall -Wextra
NVCC      ?= /usr/local/cuda/bin/nvcc
# sm_120 = RTX 5090, sm_89 = RTX 4090/Ada (needs CUDA 12.4+ for e4m3 MMA),
# sm_86 = RTX 3090 (fallback device for tests)
NVCCFLAGS ?= -O2 -std=c++17 -gencode arch=compute_86,code=sm_86 \
             -gencode arch=compute_89,code=sm_89 \
             -gencode arch=compute_120,code=sm_120 -Xcompiler -Wall

.PHONY: all clean test-inspect test-metal-backend metal-engine test-metal-contracts test-metal test-metal-canonical check-chat-extract check-responses-integration
all: build/inspect build/test_sampling build/test_kernels build/test_argmax_tie build/q27 build/q27-server build/test_tokenizer build/test_stream_split build/test_openai_bridge build/test_chat_completions_integration build/test_depthctl build/test_toolconstrain

build/q27: src/engine.cu src/engine.cuh src/blocks.cu src/prefill.cu src/kernels.cu src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp \
           src/blocks.cuh src/kernels.cuh src/spec3.cuh src/prefill.cuh src/fdmma.cuh src/turbo3.cuh src/turbo5.cuh src/device_model.h src/loader.h src/cuda_common.h src/depthctl.h src/prefix_cache.h src/prefix_ram.h | build
	$(NVCC) $(NVCCFLAGS) src/engine.cu src/blocks.cu src/prefill.cu src/kernels.cu src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp -o $@

build:
	mkdir -p build

build/inspect: src/inspect.cpp src/loader.cpp src/loader.h | build
	$(CXX) $(CXXFLAGS) src/inspect.cpp src/loader.cpp -o $@
build/test_loader_contracts: src/test_loader_contracts.cpp src/loader.cpp src/loader.h | build
	$(CXX) $(CXXFLAGS) src/test_loader_contracts.cpp src/loader.cpp -o $@


test-inspect: build/inspect build/test_loader_contracts
	python3 tools/test_inspect.py ./build/inspect
	./build/test_loader_contracts

build/test_sampling: src/test_sampling.cpp src/sampling.h | build
	$(CXX) $(CXXFLAGS) src/test_sampling.cpp -o $@

build/test_tokenizer: src/test_tokenizer.cpp src/tokenizer.cpp src/tokenizer.h src/api_common.h src/stream_split.h src/markdown_lex.h src/toolgram.h | build
	$(CXX) $(CXXFLAGS) -DQ27_TOKENIZER_TESTING src/test_tokenizer.cpp src/tokenizer.cpp -o $@

build/test_stream_split: tools/test_stream_split.cpp src/stream_split.h src/markdown_lex.h | build
	$(CXX) $(CXXFLAGS) -I src tools/test_stream_split.cpp -o $@

build/test_openai_bridge: tools/test_openai_bridge.cpp src/api_common.h src/stream_split.h src/markdown_lex.h | build
	$(CXX) $(CXXFLAGS) -I src tools/test_openai_bridge.cpp -o $@

build/test_chat_completions_integration: tools/test_chat_completions_integration.cpp src/server.cu src/api_common.h src/toolconstrain.h src/toolgram.h src/stream_split.h src/markdown_lex.h src/tokenizer.h | build
	$(CXX) $(CXXFLAGS) -I src tools/test_chat_completions_integration.cpp -o $@

build/test_auth: tools/test_auth.cpp src/api_common.h | build
	$(CXX) $(CXXFLAGS) -I src tools/test_auth.cpp -o $@

build/test_auth_integration: tools/test_auth_integration.cpp src/api_common.h third_party/httplib.h | build
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

build/test_kernels: src/test_kernels.cu src/kernels.cu src/prefill.cu src/blocks.cu src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp \
                    src/kernels.cuh src/prefill.cuh src/blocks.cuh src/spec3.cuh src/fdmma.cuh src/turbo3.cuh src/turbo5.cuh src/device_model.h src/loader.h src/cuda_common.h src/sampling.h | build
	$(NVCC) $(NVCCFLAGS) src/test_kernels.cu src/kernels.cu src/prefill.cu src/blocks.cu src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp -o $@

build/test_argmax_tie: tools/test_argmax_tie.cu src/blocks.cu src/blocks.cuh | build
	$(NVCC) $(NVCCFLAGS) tools/test_argmax_tie.cu src/blocks.cu -o $@


build/q27-server: src/server.cu src/engine.cuh src/conductor.h src/blocks.cu src/prefill.cu src/kernels.cu src/spec3.cu src/vgemm.cu \
                  src/device_model.cu src/loader.cpp src/tokenizer.cpp src/api_common.h src/stream_split.h src/markdown_lex.h \
                  src/blocks.cuh src/kernels.cuh src/spec3.cuh src/prefill.cuh src/fdmma.cuh src/turbo3.cuh src/turbo5.cuh src/cuda_common.h src/toolgram.h \
                  src/depthctl.h src/toolconstrain.h src/tokenizer.h src/prefix_cache.h src/prefix_ram.h | build
	$(NVCC) $(NVCCFLAGS) -Xcompiler -pthread src/server.cu src/blocks.cu src/prefill.cu src/kernels.cu \
	        src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp src/tokenizer.cpp -o $@

clean:
	rm -rf build

build/gdn_chunk_bench: tools/gdn_chunk_bench.cu | build
	$(NVCC) $(NVCCFLAGS) tools/gdn_chunk_bench.cu -o $@

build/attn_fdw_bench: tools/attn_fdw_bench.cu | build
	$(NVCC) $(NVCCFLAGS) tools/attn_fdw_bench.cu -o $@

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

build/fdmma_test: tools/fdmma_test.cu src/fdmma.cuh | build
	$(NVCC) $(NVCCFLAGS) tools/fdmma_test.cu -o $@

build/turbo3_test: tools/turbo3_test.cu src/turbo3.cuh | build
	$(NVCC) $(NVCCFLAGS) tools/turbo3_test.cu -o $@

# turbo5 (5-bit K) format gate, docs/plans/2026-08-01-5bit-k.md phase P0.
# Depends on turbo3.cuh too: turbo5.cuh includes it for the shared WHT.
build/turbo5_test: tools/turbo5_test.cu src/turbo5.cuh src/turbo3.cuh | build
	$(NVCC) $(NVCCFLAGS) tools/turbo5_test.cu -o $@

# 24GB-card (3090-class) server: Q27_W_MAX=8 shrinks the GDN role sets +
# graph zoo so the fixed stack fits beside the weights (the default W12
# build OOMs at graph instantiation on 24GB). Same sources, own binary.
build/q27-server-w8: src/server.cu src/engine.cuh src/conductor.h src/blocks.cu src/prefill.cu src/kernels.cu src/spec3.cu src/vgemm.cu \
                     src/device_model.cu src/loader.cpp src/tokenizer.cpp src/api_common.h src/stream_split.h src/markdown_lex.h \
                     src/blocks.cuh src/kernels.cuh src/spec3.cuh src/prefill.cuh src/fdmma.cuh src/turbo3.cuh src/turbo5.cuh src/cuda_common.h src/toolgram.h \
                     src/depthctl.h src/toolconstrain.h src/tokenizer.h src/prefix_cache.h src/prefix_ram.h | build
	$(NVCC) $(NVCCFLAGS) -DQ27_W_MAX=8 -Xcompiler -pthread src/server.cu src/blocks.cu src/prefill.cu src/kernels.cu \
	        src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp src/tokenizer.cpp -o $@

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

build/fused_smoke: tools/fused_smoke.cu src/engine.cuh src/conductor.h src/prefix_cache.h src/prefix_ram.h src/blocks.cu src/prefill.cu \
                   src/kernels.cu src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp | build
	$(NVCC) $(NVCCFLAGS) tools/fused_smoke.cu src/blocks.cu src/prefill.cu src/kernels.cu \
	        src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp -o $@

# w16 serving build (batch mode's natural target; was hand-built since part 10)
build/q27-server-w16: src/server.cu src/engine.cuh src/conductor.h src/blocks.cu src/prefill.cu src/kernels.cu src/spec3.cu src/vgemm.cu \
                      src/device_model.cu src/loader.cpp src/tokenizer.cpp src/api_common.h src/stream_split.h src/markdown_lex.h \
                      src/blocks.cuh src/kernels.cuh src/spec3.cuh src/prefill.cuh src/fdmma.cuh src/turbo3.cuh src/turbo5.cuh src/cuda_common.h src/toolgram.h \
                      src/depthctl.h src/toolconstrain.h src/tokenizer.h src/prefix_cache.h src/prefix_ram.h | build
	$(NVCC) $(NVCCFLAGS) -DQ27_W_MAX=16 -Xcompiler -pthread src/server.cu src/blocks.cu src/prefill.cu src/kernels.cu \
	        src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp src/tokenizer.cpp -o $@

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
                      src/sampling.h src/suffixdraft.h third_party/json.hpp | build
	$(CXX) $(METALFLAGS) -c src/metal/metal_engine.cpp -o $@

build/test_metal_engine_contracts: src/metal/test_metal_engine_contracts.cpp \
                                   src/metal/metal_engine.cpp src/metal/metal_engine.h \
                                   src/metal/metal_backend.mm src/metal/metal_backend.h \
                                   src/metal/q27_kernels.metal src/backend.h \
                                   src/loader.cpp src/loader.h src/sampling.h src/suffixdraft.h \
                                   third_party/json.hpp | build
	$(CXX) $(METALFLAGS) src/metal/test_metal_engine_contracts.cpp \
	       src/metal/metal_engine.cpp src/metal/metal_backend.mm src/loader.cpp \
	       $(METALLIBS) -o $@

test-metal-contracts: build/test_metal_engine_contracts
	@test -n "$(MODEL)" || { echo "set MODEL=...q4s.q27" >&2; exit 2; }
	./build/test_metal_engine_contracts "$(MODEL)"

metal-engine: build/metal-engine.o

build/q27-metal: src/metal/metal_cli.cpp src/metal/metal_engine.cpp \
                 src/metal/metal_backend.mm src/metal/metal_engine.h \
                 src/metal/metal_backend.h src/metal/q27_kernels.metal \
                 src/backend.h src/loader.cpp src/loader.h src/tokenizer.cpp \
                 src/tokenizer.h src/sampling.h src/suffixdraft.h third_party/json.hpp | build
	$(CXX) $(METALFLAGS) src/metal/metal_cli.cpp src/metal/metal_engine.cpp \
	       src/metal/metal_backend.mm src/loader.cpp src/tokenizer.cpp \
	       $(METALLIBS) -o $@

test-metal: test-metal-backend build/q27-metal
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

test-metal:
	@echo "test-metal requires macOS" >&2; exit 1

test-metal-canonical:
	@echo "test-metal-canonical requires macOS" >&2; exit 1
endif
