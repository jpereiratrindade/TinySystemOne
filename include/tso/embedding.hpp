#pragma once

#include "tensor.hpp"
#include "random.hpp"
#include "optimizer.hpp"
#include "tokenizer.hpp"
#include <vector>
#include <cstddef>
#include <cmath>
#include <algorithm>

namespace tso {

class Embedding {
public:
    Embedding(std::size_t vocab_size, std::size_t max_seq_len, std::size_t embedding_dim, Random& rng)
        : vocab_size_(vocab_size),
          max_seq_len_(max_seq_len),
          embedding_dim_(embedding_dim),
          W_tok_(vocab_size, embedding_dim),
          W_pos_(max_seq_len, embedding_dim),
          dW_tok_(vocab_size, embedding_dim),
          dW_pos_(max_seq_len, embedding_dim) {
        rng.init_he_normal(W_tok_);
        rng.init_he_normal(W_pos_);
        dW_tok_.fill(0.0f);
        dW_pos_.fill(0.0f);
    }

    // Forward pass: returns aggregated representation (e.g. Mean Pooling of token + pos embeddings)
    // Output dimension: embedding_dim_
    Vector forward_mean_pooled(const std::vector<TokenId>& tokens) {
        tokens_cache_ = tokens;
        seq_len_cache_ = std::min(tokens.size(), max_seq_len_);

        Vector pooled(embedding_dim_, 0.0f);
        if (seq_len_cache_ == 0) return pooled;

        for (std::size_t pos = 0; pos < seq_len_cache_; ++pos) {
            const auto tok_idx = static_cast<std::size_t>(tokens[pos]);
            if (tok_idx < vocab_size_) {
                for (std::size_t d = 0; d < embedding_dim_; ++d) {
                    pooled[d] += W_tok_(tok_idx, d) + W_pos_(pos, d);
                }
            }
        }

        const Scalar scale = 1.0f / static_cast<Scalar>(seq_len_cache_);
        for (auto& v : pooled) v *= scale;
        return pooled;
    }

    // Backward pass for mean-pooled representation
    void backward_mean_pooled(const Vector& grad_pooled) {
        if (seq_len_cache_ == 0) return;
        const Scalar scale = 1.0f / static_cast<Scalar>(seq_len_cache_);

        for (std::size_t pos = 0; pos < seq_len_cache_; ++pos) {
            const auto tok_idx = static_cast<std::size_t>(tokens_cache_[pos]);
            if (tok_idx < vocab_size_) {
                for (std::size_t d = 0; d < embedding_dim_; ++d) {
                    const Scalar g = grad_pooled[d] * scale;
                    dW_tok_(tok_idx, d) += g;
                    dW_pos_(pos, d) += g;
                }
            }
        }
    }

    void zero_grad() {
        dW_tok_.fill(0.0f);
        dW_pos_.fill(0.0f);
    }

    void update(AdamW& optimizer) {
        optimizer.step(W_tok_, dW_tok_, state_W_tok_);
        optimizer.step(W_pos_, dW_pos_, state_W_pos_);
    }

    [[nodiscard]] std::size_t num_params() const {
        return W_tok_.data.size() + W_pos_.data.size();
    }

    [[nodiscard]] Vector get_token_embedding(TokenId tok) const {
        const auto idx = static_cast<std::size_t>(tok);
        if (idx >= vocab_size_) return Vector(embedding_dim_, 0.0f);
        Vector vec(embedding_dim_);
        for (std::size_t d = 0; d < embedding_dim_; ++d) {
            vec[d] = W_tok_(idx, d);
        }
        return vec;
    }

    [[nodiscard]] Scalar token_cosine_similarity(TokenId tok1, TokenId tok2) const {
        Vector v1 = get_token_embedding(tok1);
        Vector v2 = get_token_embedding(tok2);

        Scalar dot = 0.0f, n1 = 0.0f, n2 = 0.0f;
        for (std::size_t d = 0; d < embedding_dim_; ++d) {
            dot += v1[d] * v2[d];
            n1 += v1[d] * v1[d];
            n2 += v2[d] * v2[d];
        }
        const Scalar denom = std::sqrt(n1) * std::sqrt(n2);
        if (denom < 1e-12f) return 0.0f;
        return dot / denom;
    }

    [[nodiscard]] std::size_t vocab_size() const { return vocab_size_; }
    [[nodiscard]] std::size_t embedding_dim() const { return embedding_dim_; }
    [[nodiscard]] std::size_t max_seq_len() const { return max_seq_len_; }
    [[nodiscard]] const Matrix& token_weights() const { return W_tok_; }
    [[nodiscard]] const Matrix& pos_weights() const { return W_pos_; }

private:
    std::size_t vocab_size_;
    std::size_t max_seq_len_;
    std::size_t embedding_dim_;

    Matrix W_tok_;
    Matrix W_pos_;
    Matrix dW_tok_;
    Matrix dW_pos_;

    ParameterState state_W_tok_;
    ParameterState state_W_pos_;

    std::vector<TokenId> tokens_cache_;
    std::size_t seq_len_cache_{0};
};

} // namespace tso
