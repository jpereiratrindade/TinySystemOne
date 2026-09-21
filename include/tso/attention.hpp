#pragma once

#include "tensor.hpp"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>
#include <iostream>
#include <iomanip>

namespace tso {

// Sequence representation: L tokens of dimension D
using Sequence = std::vector<std::vector<double>>;

/**
 * @brief Layer Normalization for D-dimensional vectors or sequences.
 *
 * Implements y = gamma * ((x - mu) / sqrt(var + eps)) + beta
 * with learnable gamma, beta and exact analytical backward pass.
 */
class LayerNorm {
public:
    size_t dim;
    double eps;

    std::vector<double> gamma;
    std::vector<double> beta;

    std::vector<double> grad_gamma;
    std::vector<double> grad_beta;

    // AdamW momentum buffers
    std::vector<double> m_gamma, v_gamma;
    std::vector<double> m_beta, v_beta;

    explicit LayerNorm(size_t d, double epsilon = 1e-5)
        : dim(d), eps(epsilon),
          gamma(d, 1.0), beta(d, 0.0),
          grad_gamma(d, 0.0), grad_beta(d, 0.0),
          m_gamma(d, 0.0), v_gamma(d, 0.0),
          m_beta(d, 0.0), v_beta(d, 0.0) {}

    // Forward single vector x -> y
    std::vector<double> forward_vector(const std::vector<double>& x,
                                       std::vector<double>& x_hat_cache,
                                       double& std_inv_cache) const {
        double mean = 0.0;
        for (double v : x) mean += v;
        mean /= static_cast<double>(dim);

        double var = 0.0;
        for (double v : x) {
            double diff = v - mean;
            var += diff * diff;
        }
        var /= static_cast<double>(dim);
        double std_inv = 1.0 / std::sqrt(var + eps);
        std_inv_cache = std_inv;

        x_hat_cache.resize(dim);
        std::vector<double> y(dim);
        for (size_t i = 0; i < dim; ++i) {
            x_hat_cache[i] = (x[i] - mean) * std_inv;
            y[i] = gamma[i] * x_hat_cache[i] + beta[i];
        }
        return y;
    }

    // Forward sequence X (L x D) -> Y (L x D)
    Sequence forward(const Sequence& x,
                     std::vector<std::vector<double>>& x_hat_cache,
                     std::vector<double>& std_inv_cache) const {
        size_t L = x.size();
        Sequence y(L);
        x_hat_cache.resize(L);
        std_inv_cache.resize(L);

        for (size_t i = 0; i < L; ++i) {
            y[i] = forward_vector(x[i], x_hat_cache[i], std_inv_cache[i]);
        }
        return y;
    }

    // Backward sequence dY -> dX, accumulating grad_gamma and grad_beta
    Sequence backward(const Sequence& dy,
                      const std::vector<std::vector<double>>& x_hat_cache,
                      const std::vector<double>& std_inv_cache) {
        size_t L = dy.size();
        Sequence dx(L, std::vector<double>(dim, 0.0));
        double N = static_cast<double>(dim);

        for (size_t t = 0; t < L; ++t) {
            const auto& cur_dy = dy[t];
            const auto& cur_x_hat = x_hat_cache[t];
            double cur_std_inv = std_inv_cache[t];

            std::vector<double> dx_hat(dim);
            double sum_dx_hat = 0.0;
            double sum_dx_hat_x_hat = 0.0;

            for (size_t i = 0; i < dim; ++i) {
                grad_gamma[i] += cur_dy[i] * cur_x_hat[i];
                grad_beta[i] += cur_dy[i];

                dx_hat[i] = cur_dy[i] * gamma[i];
                sum_dx_hat += dx_hat[i];
                sum_dx_hat_x_hat += dx_hat[i] * cur_x_hat[i];
            }

            for (size_t i = 0; i < dim; ++i) {
                dx[t][i] = (cur_std_inv / N) * (N * dx_hat[i] - sum_dx_hat - cur_x_hat[i] * sum_dx_hat_x_hat);
            }
        }
        return dx;
    }

    void zero_grad() {
        std::fill(grad_gamma.begin(), grad_gamma.end(), 0.0);
        std::fill(grad_beta.begin(), grad_beta.end(), 0.0);
    }

    void step_adamw(double lr, size_t t, double beta1 = 0.9, double beta2 = 0.999,
                    double weight_decay = 0.01, double eps_adam = 1e-8) {
        double bc1 = 1.0 - std::pow(beta1, static_cast<double>(t));
        double bc2 = 1.0 - std::pow(beta2, static_cast<double>(t));

        for (size_t i = 0; i < dim; ++i) {
            // Gamma update
            m_gamma[i] = beta1 * m_gamma[i] + (1.0 - beta1) * grad_gamma[i];
            v_gamma[i] = beta2 * v_gamma[i] + (1.0 - beta2) * grad_gamma[i] * grad_gamma[i];
            double m_hat = m_gamma[i] / bc1;
            double v_hat = v_gamma[i] / bc2;
            gamma[i] -= lr * (m_hat / (std::sqrt(v_hat) + eps_adam) + weight_decay * (gamma[i] - 1.0));

            // Beta update (no weight decay on bias)
            m_beta[i] = beta1 * m_beta[i] + (1.0 - beta1) * grad_beta[i];
            v_beta[i] = beta2 * v_beta[i] + (1.0 - beta2) * grad_beta[i] * grad_beta[i];
            m_hat = m_beta[i] / bc1;
            v_hat = v_beta[i] / bc2;
            beta[i] -= lr * (m_hat / (std::sqrt(v_hat) + eps_adam));
        }
    }
};

/**
 * @brief Multi-Head Self-Attention Layer in pure C++26.
 *
 * Implements Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
 * for h attention heads, with full analytical backward pass and AdamW.
 */
class MultiHeadAttention {
public:
    size_t d_model;
    size_t n_heads;
    size_t d_k; // d_model / n_heads

    // Projection weights: W_q, W_k, W_v, W_o (each d_model x d_model)
    // Row-major: W[r * d_model + c]
    std::vector<double> W_q, b_q;
    std::vector<double> W_k, b_k;
    std::vector<double> W_v, b_v;
    std::vector<double> W_o, b_o;

    // Gradients
    std::vector<double> dW_q, db_q;
    std::vector<double> dW_k, db_k;
    std::vector<double> dW_v, db_v;
    std::vector<double> dW_o, db_o;

    // AdamW momentum
    std::vector<double> m_Wq, v_Wq, m_bq, v_bq;
    std::vector<double> m_Wk, v_Wk, m_bk, v_bk;
    std::vector<double> m_Wv, v_Wv, m_bv, v_bv;
    std::vector<double> m_Wo, v_Wo, m_bo, v_bo;

    // Forward Cache for exact backward
    struct Cache {
        Sequence X;                              // Input sequence L x d_model
        Sequence Q, K, V;                        // Linear projections L x d_model
        std::vector<Sequence> A;                 // Attention matrices per head [n_heads][L x L]
        std::vector<Sequence> O_heads;           // Per-head context [n_heads][L x d_k]
        Sequence O_concat;                       // Concatenated context L x d_model
    };

    mutable Cache last_cache;

    MultiHeadAttention(size_t d_m, size_t heads, uint32_t seed = 42)
        : d_model(d_m), n_heads(heads), d_k(d_m / heads) {
        size_t mat_size = d_model * d_model;

        W_q.resize(mat_size); b_q.assign(d_model, 0.0);
        W_k.resize(mat_size); b_k.assign(d_model, 0.0);
        W_v.resize(mat_size); b_v.assign(d_model, 0.0);
        W_o.resize(mat_size); b_o.assign(d_model, 0.0);

        dW_q.assign(mat_size, 0.0); db_q.assign(d_model, 0.0);
        dW_k.assign(mat_size, 0.0); db_k.assign(d_model, 0.0);
        dW_v.assign(mat_size, 0.0); db_v.assign(d_model, 0.0);
        dW_o.assign(mat_size, 0.0); db_o.assign(d_model, 0.0);

        m_Wq.assign(mat_size, 0.0); v_Wq.assign(mat_size, 0.0);
        m_bq.assign(d_model, 0.0);  v_bq.assign(d_model, 0.0);
        m_Wk.assign(mat_size, 0.0); v_Wk.assign(mat_size, 0.0);
        m_bk.assign(d_model, 0.0);  v_bk.assign(d_model, 0.0);
        m_Wv.assign(mat_size, 0.0); v_Wv.assign(mat_size, 0.0);
        m_bv.assign(d_model, 0.0);  v_bv.assign(d_model, 0.0);
        m_Wo.assign(mat_size, 0.0); v_Wo.assign(mat_size, 0.0);
        m_bo.assign(d_model, 0.0);  v_bo.assign(d_model, 0.0);

        init_weights(seed);
    }

    void init_weights(uint32_t seed) {
        std::mt19937 gen(seed);
        // Xavier/Glorot initialization: std = sqrt(2 / (fan_in + fan_out))
        double std_dev = std::sqrt(2.0 / static_cast<double>(2 * d_model));
        std::normal_distribution<double> dist(0.0, std_dev);

        size_t mat_size = d_model * d_model;
        for (size_t i = 0; i < mat_size; ++i) {
            W_q[i] = dist(gen);
            W_k[i] = dist(gen);
            W_v[i] = dist(gen);
            W_o[i] = dist(gen);
        }
    }

    // Linear projection helper: Y = X * W + b
    static Sequence linear_forward(const Sequence& X, const std::vector<double>& W, const std::vector<double>& b, size_t d_in, size_t d_out) {
        size_t L = X.size();
        Sequence Y(L, std::vector<double>(d_out, 0.0));
        for (size_t i = 0; i < L; ++i) {
            for (size_t o = 0; o < d_out; ++o) {
                double sum = b[o];
                for (size_t in = 0; in < d_in; ++in) {
                    sum += X[i][in] * W[in * d_out + o];
                }
                Y[i][o] = sum;
            }
        }
        return Y;
    }

    // Forward pass
    Sequence forward(const Sequence& X) const {
        size_t L = X.size();
        last_cache.X = X;
        last_cache.Q = linear_forward(X, W_q, b_q, d_model, d_model);
        last_cache.K = linear_forward(X, W_k, b_k, d_model, d_model);
        last_cache.V = linear_forward(X, W_v, b_v, d_model, d_model);

        last_cache.A.resize(n_heads);
        last_cache.O_heads.resize(n_heads);
        last_cache.O_concat.assign(L, std::vector<double>(d_model, 0.0));

        double scale = 1.0 / std::sqrt(static_cast<double>(d_k));

        for (size_t h = 0; h < n_heads; ++h) {
            size_t head_offset = h * d_k;

            // Score matrix S: L x L
            Sequence S(L, std::vector<double>(L, 0.0));
            Sequence A(L, std::vector<double>(L, 0.0));
            Sequence O_h(L, std::vector<double>(d_k, 0.0));

            for (size_t i = 0; i < L; ++i) {
                // Compute dot-product with all keys j
                double max_score = -1e9;
                for (size_t j = 0; j < L; ++j) {
                    double score = 0.0;
                    for (size_t k = 0; k < d_k; ++k) {
                        score += last_cache.Q[i][head_offset + k] * last_cache.K[j][head_offset + k];
                    }
                    score *= scale;
                    S[i][j] = score;
                    if (score > max_score) max_score = score;
                }

                // Row-wise Softmax
                double sum_exp = 0.0;
                for (size_t j = 0; j < L; ++j) {
                    A[i][j] = std::exp(S[i][j] - max_score);
                    sum_exp += A[i][j];
                }
                double inv_sum = 1.0 / (sum_exp > 1e-12 ? sum_exp : 1e-12);
                for (size_t j = 0; j < L; ++j) {
                    A[i][j] *= inv_sum;
                }

                // Context O_h = A * V_h
                for (size_t k = 0; k < d_k; ++k) {
                    double val_sum = 0.0;
                    for (size_t j = 0; j < L; ++j) {
                        val_sum += A[i][j] * last_cache.V[j][head_offset + k];
                    }
                    O_h[i][k] = val_sum;
                    last_cache.O_concat[i][head_offset + k] = val_sum;
                }
            }

            last_cache.A[h] = std::move(A);
            last_cache.O_heads[h] = std::move(O_h);
        }

        // Final linear projection
        return linear_forward(last_cache.O_concat, W_o, b_o, d_model, d_model);
    }

    // Exact analytical backward pass: returns dX (L x d_model)
    Sequence backward(const Sequence& dY) {
        size_t L = dY.size();
        double scale = 1.0 / std::sqrt(static_cast<double>(d_k));

        // 1. Gradients for Output Projection W_o, b_o and dO_concat
        Sequence dO_concat(L, std::vector<double>(d_model, 0.0));
        for (size_t i = 0; i < L; ++i) {
            for (size_t o = 0; o < d_model; ++o) {
                db_o[o] += dY[i][o];
                for (size_t in = 0; in < d_model; ++in) {
                    dW_o[in * d_model + o] += last_cache.O_concat[i][in] * dY[i][o];
                    dO_concat[i][in] += dY[i][o] * W_o[in * d_model + o];
                }
            }
        }

        // 2. Gradients per head
        Sequence dQ(L, std::vector<double>(d_model, 0.0));
        Sequence dK(L, std::vector<double>(d_model, 0.0));
        Sequence dV(L, std::vector<double>(d_model, 0.0));

        for (size_t h = 0; h < n_heads; ++h) {
            size_t head_offset = h * d_k;
            const auto& A = last_cache.A[h];

            // dA: L x L
            Sequence dA(L, std::vector<double>(L, 0.0));

            // Backprop from O_h = A * V_h
            for (size_t i = 0; i < L; ++i) {
                for (size_t k = 0; k < d_k; ++k) {
                    double cur_dO = dO_concat[i][head_offset + k];
                    for (size_t j = 0; j < L; ++j) {
                        dA[i][j] += cur_dO * last_cache.V[j][head_offset + k];
                        dV[j][head_offset + k] += cur_dO * A[i][j];
                    }
                }
            }

            // Backprop through Softmax: dS = A * (dA - sum(dA * A))
            Sequence dS(L, std::vector<double>(L, 0.0));
            for (size_t i = 0; i < L; ++i) {
                double sum_dA_A = 0.0;
                for (size_t j = 0; j < L; ++j) {
                    sum_dA_A += dA[i][j] * A[i][j];
                }
                for (size_t j = 0; j < L; ++j) {
                    dS[i][j] = A[i][j] * (dA[i][j] - sum_dA_A) * scale;
                }
            }

            // Backprop from S = scale * Q_h * K_h^T
            for (size_t i = 0; i < L; ++i) {
                for (size_t j = 0; j < L; ++j) {
                    double cur_dS = dS[i][j];
                    for (size_t k = 0; k < d_k; ++k) {
                        dQ[i][head_offset + k] += cur_dS * last_cache.K[j][head_offset + k];
                        dK[j][head_offset + k] += cur_dS * last_cache.Q[i][head_offset + k];
                    }
                }
            }
        }

        // 3. Gradients for W_q, W_k, W_v and input dX
        Sequence dX(L, std::vector<double>(d_model, 0.0));

        for (size_t i = 0; i < L; ++i) {
            for (size_t o = 0; o < d_model; ++o) {
                db_q[o] += dQ[i][o];
                db_k[o] += dK[i][o];
                db_v[o] += dV[i][o];

                for (size_t in = 0; in < d_model; ++in) {
                    dW_q[in * d_model + o] += last_cache.X[i][in] * dQ[i][o];
                    dW_k[in * d_model + o] += last_cache.X[i][in] * dK[i][o];
                    dW_v[in * d_model + o] += last_cache.X[i][in] * dV[i][o];

                    dX[i][in] += dQ[i][o] * W_q[in * d_model + o] +
                                 dK[i][o] * W_k[in * d_model + o] +
                                 dV[i][o] * W_v[in * d_model + o];
                }
            }
        }

        return dX;
    }

    void zero_grad() {
        std::fill(dW_q.begin(), dW_q.end(), 0.0);
        std::fill(db_q.begin(), db_q.end(), 0.0);
        std::fill(dW_k.begin(), dW_k.end(), 0.0);
        std::fill(db_k.begin(), db_k.end(), 0.0);
        std::fill(dW_v.begin(), dW_v.end(), 0.0);
        std::fill(db_v.begin(), db_v.end(), 0.0);
        std::fill(dW_o.begin(), dW_o.end(), 0.0);
        std::fill(db_o.begin(), db_o.end(), 0.0);
    }

    void update_param_adamw(std::vector<double>& W, const std::vector<double>& dW,
                            std::vector<double>& m, std::vector<double>& v,
                            double lr, double bc1, double bc2,
                            double weight_decay, double eps_adam) {
        size_t n = W.size();
        for (size_t i = 0; i < n; ++i) {
            m[i] = 0.9 * m[i] + 0.1 * dW[i];
            v[i] = 0.999 * v[i] + 0.001 * dW[i] * dW[i];
            double m_hat = m[i] / bc1;
            double v_hat = v[i] / bc2;
            W[i] -= lr * (m_hat / (std::sqrt(v_hat) + eps_adam) + weight_decay * W[i]);
        }
    }

    void update_bias_adamw(std::vector<double>& b, const std::vector<double>& db,
                           std::vector<double>& m, std::vector<double>& v,
                           double lr, double bc1, double bc2,
                           double eps_adam) {
        size_t n = b.size();
        for (size_t i = 0; i < n; ++i) {
            m[i] = 0.9 * m[i] + 0.1 * db[i];
            v[i] = 0.999 * v[i] + 0.001 * db[i] * db[i];
            double m_hat = m[i] / bc1;
            double v_hat = v[i] / bc2;
            b[i] -= lr * (m_hat / (std::sqrt(v_hat) + eps_adam));
        }
    }

    void step_adamw(double lr, size_t t, double weight_decay = 0.01, double eps_adam = 1e-8) {
        double bc1 = 1.0 - std::pow(0.9, static_cast<double>(t));
        double bc2 = 1.0 - std::pow(0.999, static_cast<double>(t));

        update_param_adamw(W_q, dW_q, m_Wq, v_Wq, lr, bc1, bc2, weight_decay, eps_adam);
        update_bias_adamw(b_q, db_q, m_bq, v_bq, lr, bc1, bc2, eps_adam);

        update_param_adamw(W_k, dW_k, m_Wk, v_Wk, lr, bc1, bc2, weight_decay, eps_adam);
        update_bias_adamw(b_k, db_k, m_bk, v_bk, lr, bc1, bc2, eps_adam);

        update_param_adamw(W_v, dW_v, m_Wv, v_Wv, lr, bc1, bc2, weight_decay, eps_adam);
        update_bias_adamw(b_v, db_v, m_bv, v_bv, lr, bc1, bc2, eps_adam);

        update_param_adamw(W_o, dW_o, m_Wo, v_Wo, lr, bc1, bc2, weight_decay, eps_adam);
        update_bias_adamw(b_o, db_o, m_bo, v_bo, lr, bc1, bc2, eps_adam);
    }

    // Inspect attention probability matrix for a head [L x L]
    const Sequence& get_attention_weights(size_t head_idx) const {
        return last_cache.A[head_idx];
    }
};

} // namespace tso
