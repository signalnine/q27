// q27 container reader. mmap-based, zero-copy: tensor data pointers alias the mapping.
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace q27 {

enum class DType : uint8_t { F32 = 0, F16 = 1, Q8_G128 = 2, Q4_G64 = 3 };

const char* dtype_name(DType t);

struct Tensor {
    std::string name;
    DType dtype = DType::F32;
    std::vector<uint64_t> shape;   // row-major, contiguous axis LAST
    const uint8_t* data = nullptr; // into mmap
    uint64_t data_size = 0;
    const uint8_t* scales = nullptr; // fp16 group scales, nullptr for F32/F16
    uint64_t scales_size = 0;

    uint64_t rows() const;  // product of all dims except last (1 for 1-D)
    uint64_t cols() const;  // last dim
    uint64_t n_elements() const;
};

struct Model {
    std::string meta_json;                        // raw JSON metadata blob
    std::vector<Tensor> tensors;                  // file order
    std::unordered_map<std::string, size_t> index; // name -> tensors[] idx

    const Tensor* find(const std::string& name) const;
    const Tensor& get(const std::string& name) const; // throws if missing

    Model() = default;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&& o) noexcept;
    Model& operator=(Model&& o) noexcept;
    ~Model();

    static Model open(const std::string& path); // throws std::runtime_error

  private:
    void* map_base_ = nullptr;
    size_t map_size_ = 0;
};

} // namespace q27
