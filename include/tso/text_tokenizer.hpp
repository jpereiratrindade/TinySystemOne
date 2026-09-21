#pragma once

#include "tokenizer.hpp"
#include <string>
#include <string_view>
#include <vector>
#include <array>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <sstream>

namespace tso {

/**
 * @brief Controlled Lexicon Text Tokenizer for Semi-Structured System One domain (v0.9.0 / EXP-009).
 *
 * Implements case-insensitive tokenization with punctuation separation,
 * special tokens ([CLS], [SEP], [PAD], [UNK]), and vocabulary lookup in pure C++26.
 */
class TextTokenizer {
public:
    static constexpr std::size_t kMaxSequenceLength = 32;

    TextTokenizer() {
        init_vocabulary();
    }

    [[nodiscard]] std::size_t vocab_size() const noexcept {
        return vocab_list_.size();
    }

    [[nodiscard]] std::string_view token_to_string(std::size_t id) const {
        if (id < vocab_list_.size()) {
            return vocab_list_[id];
        }
        return "[UNK]";
    }

    [[nodiscard]] std::size_t string_to_token(std::string_view str) const {
        std::string lower_str = to_lower(str);
        auto it = vocab_map_.find(lower_str);
        if (it != vocab_map_.end()) {
            return it->second;
        }
        return unk_id_;
    }

    // Tokenizes raw text into fixed sequence of length max_len with [CLS] and [SEP]
    std::vector<TokenId> tokenize(std::string_view text, std::size_t max_len = kMaxSequenceLength) const {
        std::vector<TokenId> tokens;
        tokens.reserve(max_len);

        // 1. Always start with [CLS]
        tokens.push_back(static_cast<TokenId>(cls_id_));

        // 2. Lexical splitting on whitespace and punctuation
        std::vector<std::string> words = split_words_and_punctuation(text);

        for (const auto& w : words) {
            if (tokens.size() >= max_len - 1) break; // Reserve space for [SEP]
            std::size_t id = string_to_token(w);
            tokens.push_back(static_cast<TokenId>(id));
        }

        // 3. Append [SEP]
        tokens.push_back(static_cast<TokenId>(sep_id_));

        // 4. Pad with [PAD] up to max_len
        while (tokens.size() < max_len) {
            tokens.push_back(static_cast<TokenId>(pad_id_));
        }

        return tokens;
    }

    std::string decode(const std::vector<TokenId>& tokens, bool skip_special = false) const {
        std::ostringstream oss;
        bool first = true;
        for (auto tok : tokens) {
            auto id = static_cast<std::size_t>(tok);
            if (skip_special && (id == pad_id_ || id == cls_id_ || id == sep_id_)) {
                continue;
            }
            if (!first) oss << " ";
            oss << token_to_string(id);
            first = false;
        }
        return oss.str();
    }

    [[nodiscard]] std::size_t cls_id() const noexcept { return cls_id_; }
    [[nodiscard]] std::size_t sep_id() const noexcept { return sep_id_; }
    [[nodiscard]] std::size_t pad_id() const noexcept { return pad_id_; }
    [[nodiscard]] std::size_t unk_id() const noexcept { return unk_id_; }
    [[nodiscard]] std::size_t mask_id() const noexcept { return mask_id_; }

private:
    std::vector<std::string> vocab_list_;
    std::unordered_map<std::string, std::size_t> vocab_map_;

    std::size_t pad_id_{0};
    std::size_t unk_id_{1};
    std::size_t cls_id_{2};
    std::size_t sep_id_{3};
    std::size_t mask_id_{4};

    static std::string to_lower(std::string_view s) {
        std::string res;
        res.reserve(s.size());
        for (char c : s) {
            res.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        }
        return res;
    }

    static bool is_punct(char c) {
        return c == ':' || c == '=' || c == ';' || c == '|' || c == ',' ||
               c == '.' || c == '[' || c == ']' || c == '(' || c == ')' ||
               c == '{' || c == '}' || c == '-' || c == '_';
    }

    static std::vector<std::string> split_words_and_punctuation(std::string_view text) {
        std::vector<std::string> tokens;
        std::string current;

        for (char c : text) {
            if (std::isspace(static_cast<unsigned char>(c))) {
                if (!current.empty()) {
                    tokens.push_back(current);
                    current.clear();
                }
            } else if (is_punct(c)) {
                if (!current.empty()) {
                    tokens.push_back(current);
                    current.clear();
                }
                tokens.push_back(std::string(1, c));
            } else {
                current.push_back(c);
            }
        }
        if (!current.empty()) {
            tokens.push_back(current);
        }
        return tokens;
    }

    void add_token(std::string_view tok) {
        std::string str(tok);
        if (vocab_map_.find(str) == vocab_map_.end()) {
            std::size_t id = vocab_list_.size();
            vocab_list_.push_back(str);
            vocab_map_[str] = id;
        }
    }

    void init_vocabulary() {
        vocab_list_.clear();
        vocab_map_.clear();

        // 1. Special Tokens
        pad_id_  = 0; add_token("[pad]");
        unk_id_  = 1; add_token("[unk]");
        cls_id_  = 2; add_token("[cls]");
        sep_id_  = 3; add_token("[sep]");
        mask_id_ = 4; add_token("[mask]");

        // 2. Syntax & Structural Punctuation
        const char* puncts[] = {
            ":", "=", ";", "|", ",", ".", "[", "]", "(", ")", "{", "}", "-"
        };
        for (const auto* p : puncts) add_token(p);

        // 3. Domain Nouns & Keys
        const char* domain_nouns[] = {
            "service", "declaration", "registration", "runtime", "process",
            "witness", "freshness", "health", "signal", "telemetry",
            "status", "score", "audit", "state", "discovery", "execution",
            "verification", "metric", "condition", "locus", "evidence",
            "decl", "reg", "run", "wit", "fresh", "hlth", "sec"
        };
        for (const auto* n : domain_nouns) add_token(n);

        // 4. Values & Adjectives
        const char* domain_values[] = {
            "true", "false", "active", "inactive", "running", "absent",
            "valid", "invalid", "stale", "aging", "expired",
            "healthy", "degraded", "failing", "nominal", "inconsistent",
            "unknown", "missing", "confirmed", "faulted", "present", "none"
        };
        for (const auto* v : domain_values) add_token(v);

        // 5. Connectives, Prepositions & Fillers
        const char* connectives[] = {
            "is", "and", "but", "with", "has", "at", "by", "for", "of",
            "in", "not", "reported", "observed", "detected", "level"
        };
        for (const auto* c : connectives) add_token(c);
    }
};

} // namespace tso
