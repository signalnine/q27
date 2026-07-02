CXX       ?= g++
CXXFLAGS  ?= -O2 -std=c++17 -Wall -Wextra
NVCC      ?= /usr/local/cuda/bin/nvcc
# sm_120 = RTX 5090, sm_86 = RTX 3090 (fallback device for tests)
NVCCFLAGS ?= -O2 -std=c++17 -gencode arch=compute_86,code=sm_86 \
             -gencode arch=compute_120,code=sm_120 -Xcompiler -Wall

.PHONY: all clean
all: build/inspect build/test_kernels build/q27 build/q27-server

build/q27: src/engine.cu src/engine.cuh src/blocks.cu src/kernels.cu src/spec3.cu src/device_model.cu src/loader.cpp \
           src/blocks.cuh src/kernels.cuh src/device_model.h src/loader.h src/cuda_common.h | build
	$(NVCC) $(NVCCFLAGS) src/engine.cu src/blocks.cu src/kernels.cu src/spec3.cu src/device_model.cu src/loader.cpp -o $@

build:
	mkdir -p build

build/inspect: src/inspect.cpp src/loader.cpp src/loader.h | build
	$(CXX) $(CXXFLAGS) src/inspect.cpp src/loader.cpp -o $@

build/test_kernels: src/test_kernels.cu src/kernels.cu src/device_model.cu src/loader.cpp \
                    src/kernels.cuh src/device_model.h src/loader.h src/cuda_common.h | build
	$(NVCC) $(NVCCFLAGS) src/test_kernels.cu src/kernels.cu src/device_model.cu src/loader.cpp -o $@


build/q27-server: src/server.cu src/engine.cuh src/blocks.cu src/kernels.cu src/spec3.cu \
                  src/device_model.cu src/loader.cpp src/tokenizer.cpp | build
	$(NVCC) $(NVCCFLAGS) -Xcompiler -pthread src/server.cu src/blocks.cu src/kernels.cu \
	        src/spec3.cu src/device_model.cu src/loader.cpp src/tokenizer.cpp -o $@

clean:
	rm -rf build
