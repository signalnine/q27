#include "loader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>

namespace q27 {

static constexpr uint32_t MAGIC = 0x46373251; // "Q27F" LE
static constexpr uint32_t VERSION = 1;
static constexpr uint64_t ALIGN = 256;

const char* dtype_name(DType t) {
    switch (t) {
        case DType::F32:     return "F32";
        case DType::F16:     return "F16";
        case DType::Q8_G128: return "Q8_G128";
        case DType::Q4_G64:  return "Q4_G64";
    }
    return "?";
}

uint64_t Tensor::rows() const {
    if (shape.size() <= 1) return 1;
    uint64_t r = 1;
    for (size_t i = 0; i + 1 < shape.size(); i++) r *= shape[i];
    return r;
}
uint64_t Tensor::cols() const { return shape.empty() ? 0 : shape.back(); }
uint64_t Tensor::n_elements() const { return rows() * cols(); }

const Tensor* Model::find(const std::string& name) const {
    auto it = index.find(name);
    return it == index.end() ? nullptr : &tensors[it->second];
}
const Tensor& Model::get(const std::string& name) const {
    const Tensor* t = find(name);
    if (!t) throw std::runtime_error("q27: missing tensor: " + name);
    return *t;
}

Model::Model(Model&& o) noexcept { *this = std::move(o); }
Model& Model::operator=(Model&& o) noexcept {
    if (this != &o) {
        this->~Model();
        meta_json = std::move(o.meta_json);
        tensors = std::move(o.tensors);
        index = std::move(o.index);
        map_base_ = o.map_base_; map_size_ = o.map_size_;
        o.map_base_ = nullptr; o.map_size_ = 0;
    }
    return *this;
}
Model::~Model() {
    if (map_base_) munmap(map_base_, map_size_);
    map_base_ = nullptr;
}

namespace {
struct Cursor {
    const uint8_t* p;
    const uint8_t* end;
    template <typename T> T read() {
        if (p + sizeof(T) > end) throw std::runtime_error("q27: truncated file");
        T v; std::memcpy(&v, p, sizeof(T)); p += sizeof(T);
        return v;
    }
    void bytes(void* dst, size_t n) {
        if (p + n > end) throw std::runtime_error("q27: truncated file");
        std::memcpy(dst, p, n); p += n;
    }
};
} // namespace

Model Model::open(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("q27: cannot open " + path);
    struct stat st{};
    if (fstat(fd, &st) != 0) { close(fd); throw std::runtime_error("q27: fstat failed"); }
    size_t sz = (size_t)st.st_size;
    void* base = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (base == MAP_FAILED) throw std::runtime_error("q27: mmap failed");

    Model m;
    m.map_base_ = base;
    m.map_size_ = sz;

    const uint8_t* b = (const uint8_t*)base;
    Cursor c{b, b + sz};
    if (c.read<uint32_t>() != MAGIC)   throw std::runtime_error("q27: bad magic");
    if (c.read<uint32_t>() != VERSION) throw std::runtime_error("q27: unsupported version");
    uint32_t n_tensors = c.read<uint32_t>();
    uint32_t meta_len  = c.read<uint32_t>();
    m.meta_json.resize(meta_len);
    c.bytes(m.meta_json.data(), meta_len);

    m.tensors.reserve(n_tensors);
    for (uint32_t i = 0; i < n_tensors; i++) {
        Tensor t;
        uint16_t nl = c.read<uint16_t>();
        t.name.resize(nl);
        c.bytes(t.name.data(), nl);
        t.dtype = (DType)c.read<uint8_t>();
        uint8_t nd = c.read<uint8_t>();
        t.shape.resize(nd);
        for (uint8_t d = 0; d < nd; d++) t.shape[d] = c.read<uint64_t>();
        uint64_t doff = c.read<uint64_t>();
        t.data_size   = c.read<uint64_t>();
        uint64_t soff = c.read<uint64_t>();
        t.scales_size = c.read<uint64_t>();
        // stash offsets in pointers temporarily; fixed up after base is known
        t.data   = (const uint8_t*)(uintptr_t)doff;
        t.scales = t.scales_size ? (const uint8_t*)(uintptr_t)soff : nullptr;
        m.tensors.push_back(std::move(t));
    }
    uint64_t table_end = (uint64_t)(c.p - b);
    uint64_t data_base = (table_end + ALIGN - 1) / ALIGN * ALIGN;

    for (size_t i = 0; i < m.tensors.size(); i++) {
        Tensor& t = m.tensors[i];
        uint64_t doff = (uint64_t)(uintptr_t)t.data;
        if (data_base + doff + t.data_size > sz)
            throw std::runtime_error("q27: tensor data out of range: " + t.name);
        t.data = b + data_base + doff;
        if (t.scales) {
            uint64_t soff = (uint64_t)(uintptr_t)t.scales;
            if (data_base + soff + t.scales_size > sz)
                throw std::runtime_error("q27: tensor scales out of range: " + t.name);
            t.scales = b + data_base + soff;
        }
        m.index.emplace(t.name, i);
    }
    return m;
}

} // namespace q27
