// Byte-level BPE tokenizer (GPT-2 family): NFC, the Qwen3.6/3.8 Split regex on
// Unicode classes, byte-level BPE -- HF-exact (src/unicode_tables.h).
// Loads the q27.tok export. Pretokenizer: hand-coded scanner covering the qwen
// regex for ASCII + "non-ASCII == letter" approximation; exactness is gated
// against llama-tokenize on an English/code corpus (see test_tokenizer).
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace q27 {

class Tokenizer {
  public:
    explicit Tokenizer(const std::string& tok_path);
    ~Tokenizer();
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;

    std::vector<int> encode(const std::string& text) const; // handles special tokens
    std::string decode(const std::vector<int>& ids) const;
    std::string decode_one(int id) const;

    // Vocabulary size. Backends that build their own logit buffers need it
    // without reaching into the tokenizer's internals.
    size_t vocab_size() const { return tokens_.size(); }

    int bos() const { return bos_; }
    int eos() const { return eos_; }

    // ChatML wrapper: messages as {role, content} pairs -> prompt token ids.
    // think=false appends the empty think block (Qwen3-family
    // enable_thinking=false convention) so the model answers directly.
    std::vector<int> apply_chat_template(
        const std::vector<std::pair<std::string, std::string>>& messages,
        bool think = true) const;

    // Exact-string vocab lookup (-1 if absent). Needed for added tokens like
    // <think> that BPE merges cannot form and the special-matcher (type-3
    // controls only) does not cover.
    int token_id(const std::string& s) const;

    // Decoded raw bytes per token id for grammar-mask construction (P7).
    // Control tokens (type 3) come back EMPTY: they must never be legal
    // inside a constrained region (their marker bytes would otherwise pass
    // as string content -- e.g. <|im_end|> inside a JSON string).
    std::vector<std::string> vocab_bytes() const;

  private:
    std::vector<std::string> tokens_;   // GPT-2 byte-encoded space
    std::vector<uint8_t> types_;
    int bos_ = 0, eos_ = 0;
    // lookup structures built at load
    struct Impl;
    std::unique_ptr<Impl> impl_;

    std::vector<int> bpe_word(const std::string& word) const;

    std::vector<std::string> pretokenize(const std::string& text) const;

public:
    // NFC, the checkpoint's normalizer, applied to every span between added
    // tokens before pretokenize (as HF does: added tokens are normalized=false).
    static std::string nfc(const std::string& s);
};

#ifdef Q27_TOKENIZER_TESTING
int tokenizer_live_impls_for_test();
#endif

} // namespace q27
