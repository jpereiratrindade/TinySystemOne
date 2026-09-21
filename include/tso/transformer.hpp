#pragma once

#include "attention.hpp"
#include "tensor.hpp"
#include <vector>
#include <memory>
#include <cmath>
#include <random>

namespace tso {

/**
 * @brief Complete Transformer Encoder Block in pure C++26.
 *
 * Implements:
 *   1. X_1 = X + MHA(LayerNorm_1(X))
 *   2. X_2 = X_1 + FFN(LayerNorm_2(X_1))
 * with exact analytical backpropagation and AdamW optimization.
 */
class TransformerEncoderBlock {
public:
    size_t d_model;
    size_t d_ff;
    size_t n_heads;

    LayerNorm ln1;
    MultiHeadAttention mha;
    LayerNorm ln2;

    // Feed-forward weights: W1 (d_model x d_ff), b1 (d_ff), W2 (d_ff x d_model), b2 (d_model)
    std::vector<double> W1, b1;
    std::vector<double> W2, b2;

    std::vector<double> dW1, db1;
    std::vector<double> dW2, db2;

    std::vector<double> m_W1, v_W1, m_b1, v_b1;
    std::vector<double> m_W2, v_W2, m_b2, v_b2;

    // Cache for backward pass
    struct BlockCache {
        Sequence X_in;
        Sequence X_norm1;
        std::vector<std::vector<double>> ln1_xhat;
        std::vector<double> ln1_std_inv;

        Sequence mha_out;
        Sequence X_1;

        Sequence X_norm2;
        std::vector<std::vector<double>> ln2_xhat;
        std::vector<double> ln2_std_inv;

        Sequence ffn_pre_act; // L x d_ff
        Sequence ffn_act;     // L x d_ff (after GELU)
        Sequence ffn_out;     // L x d_model
    };

    mutable BlockCache cache;

    TransformerEncoderBlock(size_t d_m, size_t heads, size_t d_feedforward = 0, uint32_t seed = 42)
        : d_model(d_m),
          d_ff(d_feedforward == 0 ? 2 * d_m : d_feedforward),
          n_heads(heads),
          ln1(d_m),
          mha(d_m, heads, seed),
          ln2(d_m) {

        W1.resize(d_model * d_ff); b1.assign(d_ff, 0.0);
        W2.resize(d_ff * d_model); b2.assign(d_model, 0.0);

        dW1.assign(d_model * d_ff, 0.0); db1.assign(d_ff, 0.0);
        dW2.assign(d_ff * d_model, 0.0); db2.assign(d_model, 0.0);

        m_W1.assign(d_model * d_ff, 0.0); v_W1.assign(d_model * d_ff, 0.0);
        m_b1.assign(d_ff, 0.0);           v_b1.assign(d_ff, 0.0);
        m_W2.assign(d_ff * d_model, 0.0); v_W2.assign(d_ff * d_model, 0.0);
        m_b2.assign(d_model, 0.0);        v_b2.assign(d_model, 0.0);

        init_ffn_weights(seed + 101);
    }

    void init_ffn_weights(uint32_t seed) {
        std::mt19937 gen(seed);
        double std1 = std::sqrt(2.0 / static_cast<double>(d_model + d_ff));
        double std2 = std::sqrt(2.0 / static_cast<double>(d_ff + d_model));
        std::normal_distribution<double> dist1(0.0, std1);
        std::normal_distribution<double> dist2(0.0, std2);

        for (auto& w : W1) w = dist1(gen);
        for (auto& w : W2) w = dist2(gen);
    }

    // Forward pass
    Sequence forward(const Sequence& X) const {
        size_t L = X.size();
        cache.X_in = X;

        // 1. Pre-LN MHA branch
        cache.X_norm1 = ln1.forward(X, cache.ln1_xhat, cache.ln1_std_inv);
        cache.mha_out = mha.forward(cache.X_norm1);

        cache.X_1.resize(L, std::vector<double>(d_model, 0.0));
        for (size_t i = 0; i < L; ++i) {
            for (size_t d = 0; d < d_model; ++d) {
                cache.X_1[i][d] = X[i][d] + cache.mha_out[i][d]; // Residual 1
            }
        }

        // 2. Pre-LN FFN branch
        cache.X_norm2 = ln2.forward(cache.X_1, cache.ln2_xhat, cache.ln2_std_inv);

        // FFN: Linear1 -> GELU -> Linear2
        cache.ffn_pre_act.assign(L, std::vector<double>(d_ff, 0.0));
        cache.ffn_act.assign(L, std::vector<double>(d_ff, 0.0));
        cache.ffn_out.assign(L, std::vector<double>(d_model, 0.0));

        for (size_t i = 0; i < L; ++i) {
            // Linear 1: d_model -> d_ff
            for (size_t f = 0; f < d_ff; ++f) {
                double val = b1[f];
                for (size_t d = 0; d < d_model; ++d) {
                    val += cache.X_norm2[i][d] * W1[d * d_ff + f];
                }
                cache.ffn_pre_act[i][f] = val;
                cache.ffn_act[i][f] = static_cast<double>(gelu(static_cast<Scalar>(val)));
            }

            // Linear 2: d_ff -> d_model
            for (size_t d = 0; d < d_model; ++d) {
                double val = b2[d];
                for (size_t f = 0; f < d_ff; ++f) {
                    val += cache.ffn_act[i][f] * W2[f * d_model + d];
                }
                cache.ffn_out[i][d] = val;
            }
        }

        // Residual 2
        Sequence X_2(L, std::vector<double>(d_model, 0.0));
        for (size_t i = 0; i < L; ++i) {
            for (size_t d = 0; d < d_model; ++d) {
                X_2[i][d] = cache.X_1[i][d] + cache.ffn_out[i][d];
            }
        }

        return X_2;
    }

    // Analytical backward pass: returns dX_in (L x d_model)
    Sequence backward(const Sequence& dX_2) {
        size_t L = dX_2.size();

        // 1. Residual 2 split: dX_1 = dX_2, dFFN_out = dX_2
        Sequence dX_1 = dX_2;
        Sequence dFFN_out = dX_2;

        // 2. Backprop through FFN: Linear2
        Sequence dFFN_act(L, std::vector<double>(d_ff, 0.0));
        for (size_t i = 0; i < L; ++i) {
            for (size_t d = 0; d < d_model; ++d) {
                db2[d] += dFFN_out[i][d];
                for (size_t f = 0; f < d_ff; ++f) {
                    dW2[f * d_model + d] += cache.ffn_act[i][f] * dFFN_out[i][d];
                    dFFN_act[i][f] += dFFN_out[i][d] * W2[f * d_model + d];
                }
            }
        }

        // 3. Backprop through GELU
        Sequence dFFN_pre_act(L, std::vector<double>(d_ff, 0.0));
        for (size_t i = 0; i < L; ++i) {
            for (size_t f = 0; f < d_ff; ++f) {
                dFFN_pre_act[i][f] = dFFN_act[i][f] * static_cast<double>(dgelu(static_cast<Scalar>(cache.ffn_pre_act[i][f])));
            }
        }

        // 4. Backprop through Linear1
        Sequence dX_norm2(L, std::vector<double>(d_model, 0.0));
        for (size_t i = 0; i < L; ++i) {
            for (size_t f = 0; f < d_ff; ++f) {
                db1[f] += dFFN_pre_act[i][f];
                for (size_t d = 0; d < d_model; ++d) {
                    dW1[d * d_ff + f] += cache.X_norm2[i][d] * dFFN_pre_act[i][f];
                    dX_norm2[i][d] += dFFN_pre_act[i][f] * W1[d * d_ff + f];
                }
            }
        }

        // 5. Backprop through LN2
        Sequence dX_1_from_ln2 = ln2.backward(dX_norm2, cache.ln2_xhat, cache.ln2_std_inv);
        for (size_t i = 0; i < L; ++i) {
            for (size_t d = 0; d < d_model; ++d) {
                dX_1[i][d] += dX_1_from_ln2[i][d];
            }
        }

        // 6. Residual 1 split: dX_in = dX_1, dMHA_out = dX_1
        Sequence dX_in = dX_1;
        Sequence dMHA_out = dX_1;

        // 7. Backprop through MHA
        Sequence dX_norm1 = mha.backward(dMHA_out);

        // 8. Backprop through LN1
        Sequence dX_in_from_ln1 = ln1.backward(dX_norm1, cache.ln1_xhat, cache.ln1_std_inv);
        for (size_t i = 0; i < L; ++i) {
            for (size_t d = 0; d < d_model; ++d) {
                dX_in[i][d] += dX_in_from_ln1[i][d];
            }
        }

        return dX_in;
    }

    void zero_grad() {
        ln1.zero_grad();
        mha.zero_grad();
        ln2.zero_grad();
        std::fill(dW1.begin(), dW1.end(), 0.0);
        std::fill(db1.begin(), db1.end(), 0.0);
        std::fill(dW2.begin(), dW2.end(), 0.0);
        std::fill(db2.begin(), db2.end(), 0.0);
    }

    void step_adamw(double lr, size_t t, double weight_decay = 0.01, double eps_adam = 1e-8) {
        ln1.step_adamw(lr, t, 0.9, 0.999, weight_decay, eps_adam);
        mha.step_adamw(lr, t, weight_decay, eps_adam);
        ln2.step_adamw(lr, t, 0.9, 0.999, weight_decay, eps_adam);

        double bc1 = 1.0 - std::pow(0.9, static_cast<double>(t));
        double bc2 = 1.0 - std::pow(0.999, static_cast<double>(t));

        mha.update_param_adamw(W1, dW1, m_W1, v_W1, lr, bc1, bc2, weight_decay, eps_adam);
        mha.update_bias_adamw(b1, db1, m_b1, v_b1, lr, bc1, bc2, eps_adam);

        mha.update_param_adamw(W2, dW2, m_W2, v_W2, lr, bc1, bc2, weight_decay, eps_adam);
        mha.update_bias_adamw(b2, db2, m_b2, v_b2, lr, bc1, bc2, eps_adam);
    }
};

} // namespace tso
