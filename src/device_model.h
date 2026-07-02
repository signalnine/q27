// Upload q27 tensors to a CUDA device. Supports selective upload so tests can
// run while another process holds most of the GPU.
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "loader.h"

namespace q27 {

struct DevTensor {
    DType dtype = DType::F32;
    uint64_t rows = 0, cols = 0;
    void* data = nullptr;   // device
    void* scales = nullptr; // device, nullptr if none
    uint64_t data_bytes = 0, scales_bytes = 0;
};

class DeviceModel {
  public:
    explicit DeviceModel(const Model& m) : model_(m) {}
    ~DeviceModel();

    // Upload one tensor (no-op if already resident). Returns the device tensor.
    const DevTensor& upload(const std::string& name);
    // Upload everything (engine path).
    void upload_all();

    const DevTensor& get(const std::string& name) const;
    bool model_has(const std::string& name) const { return model_.find(name) != nullptr; }
    size_t bytes_resident() const { return bytes_; }

    // Weight-integrity checksums: order-independent u64 word-sums of every
    // resident tensor, taken once after upload. checksum_verify() recomputes
    // and reports any drift -- detects OC/heat bit flips in resident weights,
    // which token-identity gates cannot see (they compare against the same
    // corrupted state). ~10 ms for the full 16.75 GB model.
    void checksum_baseline();
    int checksum_verify(bool print) const; // returns number of mismatched tensors

  private:
    const Model& model_;
    std::unordered_map<std::string, DevTensor> dev_;
    std::unordered_map<std::string, unsigned long long> sums_;
    size_t bytes_ = 0;
};

} // namespace q27
