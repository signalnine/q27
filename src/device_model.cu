#include <algorithm>
#include <cstdio>
#include <stdexcept>

#include "cuda_common.h"
#include "device_model.h"

namespace q27 {

DeviceModel::~DeviceModel() {
    for (auto& [k, t] : dev_) {
        if (t.data) cudaFree(t.data);
        if (t.scales) cudaFree(t.scales);
    }
}

const DevTensor& DeviceModel::upload(const std::string& name) {
    auto it = dev_.find(name);
    if (it != dev_.end()) return it->second;

    const Tensor& src = model_.get(name);
    DevTensor d;
    d.dtype = src.dtype;
    d.rows = src.rows();
    d.cols = src.cols();
    CUDA_CHECK(cudaMalloc(&d.data, src.data_size));
    CUDA_CHECK(cudaMemcpy(d.data, src.data, src.data_size, cudaMemcpyHostToDevice));
    bytes_ += src.data_size;
    d.data_bytes = src.data_size;
    if (src.scales) {
        CUDA_CHECK(cudaMalloc(&d.scales, src.scales_size));
        CUDA_CHECK(cudaMemcpy(d.scales, src.scales, src.scales_size, cudaMemcpyHostToDevice));
        bytes_ += src.scales_size;
        d.scales_bytes = src.scales_size;
    }
    return dev_.emplace(name, d).first->second;
}

// Order-independent u64 word-sum (wraparound add): any flipped bit changes it,
// and atomicAdd accumulation order does not.
__global__ void k_xsum64(const unsigned long long* __restrict__ p, size_t n64,
                         const unsigned char* __restrict__ tail, int ntail,
                         unsigned long long* out) {
    unsigned long long s = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n64;
         i += (size_t)gridDim.x * blockDim.x)
        s += p[i];
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffff, s, off);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, s);
    if (blockIdx.x == 0 && threadIdx.x == 0 && ntail) {
        unsigned long long t = 0;
        for (int i = 0; i < ntail; i++) t |= (unsigned long long)tail[i] << (8 * i);
        atomicAdd(out, t);
    }
}

static unsigned long long xsum_dev(const void* p, uint64_t bytes, unsigned long long* d_out) {
    CUDA_CHECK(cudaMemset(d_out, 0, 8));
    size_t n64 = bytes / 8;
    int ntail = (int)(bytes % 8);
    const unsigned char* tail = (const unsigned char*)p + n64 * 8;
    unsigned blocks = (unsigned)std::min<size_t>(4096, (n64 + 255) / 256);
    if (!blocks) blocks = 1;
    k_xsum64<<<blocks, 256>>>((const unsigned long long*)p, n64, tail, ntail, d_out);
    CUDA_CHECK(cudaGetLastError());
    unsigned long long h = 0;
    CUDA_CHECK(cudaMemcpy(&h, d_out, 8, cudaMemcpyDeviceToHost));
    return h;
}

void DeviceModel::checksum_baseline() {
    unsigned long long* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, 8));
    for (const auto& [name, t] : dev_) {
        unsigned long long s = xsum_dev(t.data, t.data_bytes, d_out);
        if (t.scales)
            s ^= 0x9e3779b97f4a7c15ULL + xsum_dev(t.scales, t.scales_bytes, d_out);
        sums_[name] = s;
    }
    CUDA_CHECK(cudaFree(d_out));
}

int DeviceModel::checksum_verify(bool print) const {
    unsigned long long* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, 8));
    int bad = 0;
    for (const auto& [name, t] : dev_) {
        auto it = sums_.find(name);
        if (it == sums_.end()) continue;
        unsigned long long s = xsum_dev(t.data, t.data_bytes, d_out);
        if (t.scales)
            s ^= 0x9e3779b97f4a7c15ULL + xsum_dev(t.scales, t.scales_bytes, d_out);
        if (s != it->second) {
            bad++;
            if (print)
                fprintf(stderr, "WEIGHT CHECKSUM MISMATCH: %s (%llx != %llx)\n", name.c_str(),
                        s, it->second);
        }
    }
    CUDA_CHECK(cudaFree(d_out));
    if (print)
        fprintf(stderr, "weight verify: %zu tensors, %d mismatched%s\n", sums_.size(), bad,
                bad ? " -- RESIDENT WEIGHTS CORRUPTED (reload required)" : "");
    return bad;
}

void DeviceModel::upload_all() {
    for (const auto& t : model_.tensors) upload(t.name);
}

const DevTensor& DeviceModel::get(const std::string& name) const {
    auto it = dev_.find(name);
    if (it == dev_.end()) throw std::runtime_error("not resident on device: " + name);
    return it->second;
}

} // namespace q27
