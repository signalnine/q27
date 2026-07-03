// Byte-level BPE tokenizer (GPT-2 family, qwen35 pretokenizer approximation).
// Loads the q27.tok export. Pretokenizer: hand-coded scanner covering the qwen
// regex for ASCII + "non-ASCII == letter" approximation; exactness is gated
// against llama-tokenize on an English/code corpus (see test_tokenizer).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace q27 {

class Tokenizer {
  public:
    explicit Tokenizer(const std::string& tok_path);

    std::vector<int> encode(const std::string& text) const; // handles special tokens
    std::string decode(const std::vector<int>& ids) const;
    std::string decode_one(int id) const;

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

  private:
    std::vector<std::string> tokens_;   // GPT-2 byte-encoded space
    std::vector<uint8_t> types_;
    int bos_ = 0, eos_ = 0;
    // lookup structures built at load
    struct Impl;
    Impl* impl_;

    std::vector<int> bpe_word(const std::string& word) const;
    std::vector<std::string> pretokenize(const std::string& text) const;
};

} // namespace q27
