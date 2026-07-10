CXX       ?= g++
CXXFLAGS  ?= -O2 -std=c++17 -Wall -Wextra
NVCC      ?= /usr/local/cuda/bin/nvcc
# sm_120 = RTX 5090, sm_86 = RTX 3090 (fallback device for tests)
NVCCFLAGS ?= -O2 -std=c++17 -gencode arch=compute_86,code=sm_86 \
             -gencode arch=compute_120,code=sm_120 -Xcompiler -Wall

.PHONY: all clean
all: build/inspect build/test_kernels build/q27 build/q27-server build/test_tokenizer build/test_depthctl build/test_toolconstrain

build/q27: src/engine.cu src/engine.cuh src/blocks.cu src/prefill.cu src/kernels.cu src/spec3.cu src/device_model.cu src/loader.cpp \
           src/blocks.cuh src/kernels.cuh src/spec3.cuh src/prefill.cuh src/device_model.h src/loader.h src/cuda_common.h src/depthctl.h | build
	$(NVCC) $(NVCCFLAGS) src/engine.cu src/blocks.cu src/prefill.cu src/kernels.cu src/spec3.cu src/device_model.cu src/loader.cpp -o $@

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

build/width_bench: tools/width_bench.cu src/kernels.cu src/spec3.cu src/blocks.cu src/prefill.cu src/device_model.cu src/loader.cpp | build
	$(NVCC) $(NVCCFLAGS) tools/width_bench.cu src/kernels.cu src/spec3.cu src/blocks.cu src/prefill.cu src/device_model.cu src/loader.cpp -o $@

build/mma16_bench: tools/mma16_bench.cu src/kernels.cu src/device_model.cu src/loader.cpp | build
	$(NVCC) $(NVCCFLAGS) tools/mma16_bench.cu src/kernels.cu src/device_model.cu src/loader.cpp -o $@

build/test_kernels: src/test_kernels.cu src/kernels.cu src/prefill.cu src/blocks.cu src/spec3.cu src/device_model.cu src/loader.cpp \
                    src/kernels.cuh src/prefill.cuh src/blocks.cuh src/spec3.cuh src/device_model.h src/loader.h src/cuda_common.h | build
	$(NVCC) $(NVCCFLAGS) src/test_kernels.cu src/kernels.cu src/prefill.cu src/blocks.cu src/spec3.cu src/device_model.cu src/loader.cpp -o $@


build/q27-server: src/server.cu src/engine.cuh src/blocks.cu src/prefill.cu src/kernels.cu src/spec3.cu \
                  src/device_model.cu src/loader.cpp src/tokenizer.cpp src/api_common.h src/stream_split.h \
                  src/blocks.cuh src/kernels.cuh src/spec3.cuh src/prefill.cuh src/cuda_common.h src/toolgram.h \
                  src/depthctl.h src/toolconstrain.h src/tokenizer.h | build
	$(NVCC) $(NVCCFLAGS) -Xcompiler -pthread src/server.cu src/blocks.cu src/prefill.cu src/kernels.cu \
	        src/spec3.cu src/device_model.cu src/loader.cpp src/tokenizer.cpp -o $@

clean:
	rm -rf build

build/gdn_chunk_bench: tools/gdn_chunk_bench.cu | build
	$(NVCC) $(NVCCFLAGS) tools/gdn_chunk_bench.cu -o $@

build/attn_fdw_bench: tools/attn_fdw_bench.cu | build
	$(NVCC) $(NVCCFLAGS) tools/attn_fdw_bench.cu -o $@

build/fdmma_test: tools/fdmma_test.cu src/fdmma.cuh | build
	$(NVCC) $(NVCCFLAGS) tools/fdmma_test.cu -o $@
