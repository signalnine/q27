CXX       ?= g++
CXXFLAGS  ?= -O2 -std=c++17 -Wall -Wextra
NVCC      ?= /usr/local/cuda/bin/nvcc
# sm_120 = RTX 5090, sm_86 = RTX 3090 (fallback device for tests)
NVCCFLAGS ?= -O2 -std=c++17 -gencode arch=compute_86,code=sm_86 \
             -gencode arch=compute_120,code=sm_120 -Xcompiler -Wall

.PHONY: all clean
all: build/inspect build/test_kernels build/q27 build/q27-server build/test_tokenizer build/test_depthctl build/test_toolconstrain

build/q27: src/engine.cu src/engine.cuh src/blocks.cu src/prefill.cu src/kernels.cu src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp \
           src/blocks.cuh src/kernels.cuh src/spec3.cuh src/prefill.cuh src/fdmma.cuh src/turbo3.cuh src/device_model.h src/loader.h src/cuda_common.h src/depthctl.h | build
	$(NVCC) $(NVCCFLAGS) src/engine.cu src/blocks.cu src/prefill.cu src/kernels.cu src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp -o $@

build:
	mkdir -p build

build/inspect: src/inspect.cpp src/loader.cpp src/loader.h | build
	$(CXX) $(CXXFLAGS) src/inspect.cpp src/loader.cpp -o $@

build/test_tokenizer: src/test_tokenizer.cpp src/tokenizer.cpp src/tokenizer.h src/api_common.h src/stream_split.h src/toolgram.h | build
	$(CXX) $(CXXFLAGS) src/test_tokenizer.cpp src/tokenizer.cpp -o $@

build/test_depthctl: tools/test_depthctl.cpp src/depthctl.h | build
	$(CXX) $(CXXFLAGS) tools/test_depthctl.cpp -o $@

build/test_toolconstrain: tools/test_toolconstrain.cpp src/toolconstrain.h src/toolgram.h | build
	$(CXX) $(CXXFLAGS) -I src tools/test_toolconstrain.cpp -o $@

build/test_suffixdraft: tools/test_suffixdraft.cpp src/suffixdraft.h | build
	$(CXX) $(CXXFLAGS) -I src tools/test_suffixdraft.cpp -o $@

build/width_bench: tools/width_bench.cu src/kernels.cu src/spec3.cu src/vgemm.cu src/blocks.cu src/prefill.cu src/device_model.cu src/loader.cpp | build
	$(NVCC) $(NVCCFLAGS) tools/width_bench.cu src/kernels.cu src/spec3.cu src/vgemm.cu src/blocks.cu src/prefill.cu src/device_model.cu src/loader.cpp -o $@

build/mma16_bench: tools/mma16_bench.cu src/kernels.cu src/device_model.cu src/loader.cpp | build
	$(NVCC) $(NVCCFLAGS) tools/mma16_bench.cu src/kernels.cu src/device_model.cu src/loader.cpp -o $@

build/test_kernels: src/test_kernels.cu src/kernels.cu src/prefill.cu src/blocks.cu src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp \
                    src/kernels.cuh src/prefill.cuh src/blocks.cuh src/spec3.cuh src/fdmma.cuh src/turbo3.cuh src/device_model.h src/loader.h src/cuda_common.h | build
	$(NVCC) $(NVCCFLAGS) src/test_kernels.cu src/kernels.cu src/prefill.cu src/blocks.cu src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp -o $@


build/q27-server: src/server.cu src/engine.cuh src/conductor.h src/blocks.cu src/prefill.cu src/kernels.cu src/spec3.cu src/vgemm.cu \
                  src/device_model.cu src/loader.cpp src/tokenizer.cpp src/api_common.h src/stream_split.h \
                  src/blocks.cuh src/kernels.cuh src/spec3.cuh src/prefill.cuh src/fdmma.cuh src/turbo3.cuh src/cuda_common.h src/toolgram.h \
                  src/depthctl.h src/toolconstrain.h src/tokenizer.h | build
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

# 24GB-card (3090-class) server: Q27_W_MAX=8 shrinks the GDN role sets +
# graph zoo so the fixed stack fits beside the weights (the default W12
# build OOMs at graph instantiation on 24GB). Same sources, own binary.
build/q27-server-w8: src/server.cu src/engine.cuh src/blocks.cu src/prefill.cu src/kernels.cu src/spec3.cu src/vgemm.cu \
                     src/device_model.cu src/loader.cpp src/tokenizer.cpp src/api_common.h src/stream_split.h \
                     src/blocks.cuh src/kernels.cuh src/spec3.cuh src/prefill.cuh src/fdmma.cuh src/turbo3.cuh src/cuda_common.h src/toolgram.h \
                     src/depthctl.h src/toolconstrain.h src/tokenizer.h | build
	$(NVCC) $(NVCCFLAGS) -DQ27_W_MAX=8 -Xcompiler -pthread src/server.cu src/blocks.cu src/prefill.cu src/kernels.cu \
	        src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp src/tokenizer.cpp -o $@

# Continuous-batching gates (docs/plans/2026-07-14-continuous-batching.md):
#   ninv_test      -- N-invariance: per-lane weight-kernel output must be bitwise
#                     independent of union width and slot (the batching contract).
#   test_conductor -- CPU: trim policy + ConductorCore membership/round-boundary.
#   fused_smoke    -- 2-engine fused round vs solo byte-identity + conductor +
#                     A2 error-injection legs (needs the GPU + model).
build/ninv_test: tools/ninv_test.cu src/vgemm.cuh src/kernels.cuh $(VGEMM_SRC) | build
	$(NVCC) $(NVCCFLAGS) tools/ninv_test.cu $(VGEMM_SRC) -o $@

build/test_conductor: tools/test_conductor.cpp src/conductor.h | build
	$(CXX) $(CXXFLAGS) -I src tools/test_conductor.cpp -o $@

build/fused_smoke: tools/fused_smoke.cu src/engine.cuh src/conductor.h src/blocks.cu src/prefill.cu \
                   src/kernels.cu src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp | build
	$(NVCC) $(NVCCFLAGS) tools/fused_smoke.cu src/blocks.cu src/prefill.cu src/kernels.cu \
	        src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp -o $@

# w16 serving build (batch mode's natural target; was hand-built since part 10)
build/q27-server-w16: src/server.cu src/engine.cuh src/conductor.h src/blocks.cu src/prefill.cu src/kernels.cu src/spec3.cu src/vgemm.cu \
                      src/device_model.cu src/loader.cpp src/tokenizer.cpp src/api_common.h src/stream_split.h \
                      src/blocks.cuh src/kernels.cuh src/spec3.cuh src/prefill.cuh src/fdmma.cuh src/turbo3.cuh src/cuda_common.h src/toolgram.h \
                      src/depthctl.h src/toolconstrain.h src/tokenizer.h | build
	$(NVCC) $(NVCCFLAGS) -DQ27_W_MAX=16 -Xcompiler -pthread src/server.cu src/blocks.cu src/prefill.cu src/kernels.cu \
	        src/spec3.cu src/vgemm.cu src/device_model.cu src/loader.cpp src/tokenizer.cpp -o $@
