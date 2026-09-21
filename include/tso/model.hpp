#pragma once

#include "tensor.hpp"
#include "random.hpp"
#include "optimizer.hpp"
#include <vector>
#include <cstddef>
#include <numeric>

namespace tso {

struct LayerConfig {
    std::size_t in_features;
    std::size_t out_features;
    Activation activation{Activation::None};
};

class Layer {
public:
    Layer(std::size_t in_features, std::size_t out_features, Activation act, Random& rng)
        : W(out_features, in_features),
          b(out_features, 0.0f),
          dW(out_features, in_features),
          db(out_features, 0.0f),
          activation(act) {
        if (act == Activation::ReLU || act == Activation::GELU) {
            rng.init_he_normal(W);
        } else {
            rng.init_xavier_uniform(W);
        }
        rng.init_zeros(b);
    }

    Vector forward(const Vector& x) {
        input_cache = x;
        pre_act_cache = matmul(W, x);
        vec_add_(pre_act_cache, b);
        output_cache = apply_activation(pre_act_cache, activation);
        return output_cache;
    }

    Vector backward(const Vector& grad_out) {
        Vector grad_pre = backward_activation(grad_out, pre_act_cache, activation);
        outer_product_accumulate(dW, grad_pre, input_cache);
        vec_add_(db, grad_pre);
        return matmul_transpose(W, grad_pre);
    }

    void zero_grad() {
        dW.fill(0.0f);
        std::fill(db.begin(), db.end(), 0.0f);
    }

    void update(AdamW& optimizer) {
        optimizer.step(W, dW, state_W);
        optimizer.step_vector(b, db, state_b);
    }

    [[nodiscard]] std::size_t num_params() const {
        return W.data.size() + b.size();
    }

    Matrix W;
    Vector b;
    Matrix dW;
    Vector db;
    Activation activation;

    ParameterState state_W;
    ParameterState state_b;

    Vector input_cache;
    Vector pre_act_cache;
    Vector output_cache;
};

class MLP {
public:
    MLP(const std::vector<LayerConfig>& configs, Random& rng) {
        layers_.reserve(configs.size());
        for (const auto& cfg : configs) {
            layers_.emplace_back(cfg.in_features, cfg.out_features, cfg.activation, rng);
        }
    }

    // Forward pass returning normalized probability distribution (Softmax)
    Vector forward(const Vector& x) {
        Vector current = x;
        for (auto& layer : layers_) {
            current = layer.forward(current);
        }
        logits_cache_ = current;
        probs_cache_ = softmax(logits_cache_);
        return probs_cache_;
    }

    // Backward pass starting from dlogits = (probs - targets)
    void backward(const Vector& dlogits) {
        Vector grad = dlogits;
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            grad = it->backward(grad);
        }
    }

    void zero_grad() {
        for (auto& layer : layers_) {
            layer.zero_grad();
        }
    }

    void update(AdamW& optimizer) {
        for (auto& layer : layers_) {
            layer.update(optimizer);
        }
    }

    [[nodiscard]] const Vector& logits() const { return logits_cache_; }
    [[nodiscard]] const Vector& probs() const { return probs_cache_; }

    [[nodiscard]] std::size_t num_params() const {
        std::size_t total = 0;
        for (const auto& layer : layers_) {
            total += layer.num_params();
        }
        return total;
    }

    [[nodiscard]] std::vector<Layer>& layers() { return layers_; }
    [[nodiscard]] const std::vector<Layer>& layers() const { return layers_; }

private:
    std::vector<Layer> layers_;
    Vector logits_cache_;
    Vector probs_cache_;
};

struct MultiHeadOutput {
    Vector choice_probs;
    Vector noul_probs;
    Scalar score{0.0f};
    Scalar uncertainty{0.0f};
};

class MultiHeadMLP {
public:
    MultiHeadMLP(
        const std::vector<LayerConfig>& trunk_configs,
        std::size_t latent_dim,
        std::size_t choice_dim,
        std::size_t noul_dim,
        Random& rng
    ) : choice_head_(latent_dim, choice_dim, Activation::None, rng),
        noul_head_(latent_dim, noul_dim, Activation::None, rng),
        score_head_(latent_dim, 1, Activation::Sigmoid, rng),
        uncertainty_head_(latent_dim, 1, Activation::Softplus, rng) {
        trunk_.reserve(trunk_configs.size());
        for (const auto& cfg : trunk_configs) {
            trunk_.emplace_back(cfg.in_features, cfg.out_features, cfg.activation, rng);
        }
    }

    MultiHeadOutput forward(const Vector& x) {
        Vector h = x;
        for (auto& layer : trunk_) {
            h = layer.forward(h);
        }

        Vector c_logits = choice_head_.forward(h);
        Vector n_logits = noul_head_.forward(h);
        Vector s_out = score_head_.forward(h);
        Vector u_out = uncertainty_head_.forward(h);

        choice_probs_cache_ = softmax(c_logits);
        noul_probs_cache_ = softmax(n_logits);
        score_cache_ = s_out[0];
        uncertainty_cache_ = u_out[0];

        return MultiHeadOutput{
            .choice_probs = choice_probs_cache_,
            .noul_probs = noul_probs_cache_,
            .score = score_cache_,
            .uncertainty = uncertainty_cache_
        };
    }

    void backward(
        const Vector& dlogits_choice,
        const Vector& dlogits_noul,
        const Vector& dscore,
        const Vector& duncertainty
    ) {
        Vector d_h_choice = choice_head_.backward(dlogits_choice);
        Vector d_h_noul = noul_head_.backward(dlogits_noul);
        Vector d_h_score = score_head_.backward(dscore);
        Vector d_h_unc = uncertainty_head_.backward(duncertainty);

        // Sum gradient contributions at trunk bottleneck
        Vector d_h = std::move(d_h_choice);
        vec_add_(d_h, d_h_noul);
        vec_add_(d_h, d_h_score);
        vec_add_(d_h, d_h_unc);

        Vector grad = d_h;
        for (auto it = trunk_.rbegin(); it != trunk_.rend(); ++it) {
            grad = it->backward(grad);
        }
    }

    void zero_grad() {
        for (auto& layer : trunk_) {
            layer.zero_grad();
        }
        choice_head_.zero_grad();
        noul_head_.zero_grad();
        score_head_.zero_grad();
        uncertainty_head_.zero_grad();
    }

    void update(AdamW& optimizer) {
        for (auto& layer : trunk_) {
            layer.update(optimizer);
        }
        choice_head_.update(optimizer);
        noul_head_.update(optimizer);
        score_head_.update(optimizer);
        uncertainty_head_.update(optimizer);
    }

    [[nodiscard]] std::size_t num_params() const {
        std::size_t total = 0;
        for (const auto& layer : trunk_) {
            total += layer.num_params();
        }
        total += choice_head_.num_params();
        total += noul_head_.num_params();
        total += score_head_.num_params();
        total += uncertainty_head_.num_params();
        return total;
    }

    [[nodiscard]] std::vector<Layer>& trunk() { return trunk_; }
    [[nodiscard]] Layer& choice_head() { return choice_head_; }
    [[nodiscard]] Layer& noul_head() { return noul_head_; }
    [[nodiscard]] Layer& score_head() { return score_head_; }
    [[nodiscard]] Layer& uncertainty_head() { return uncertainty_head_; }

private:
    std::vector<Layer> trunk_;
    Layer choice_head_;
    Layer noul_head_;
    Layer score_head_;
    Layer uncertainty_head_;

    Vector choice_probs_cache_;
    Vector noul_probs_cache_;
    Scalar score_cache_{0.0f};
    Scalar uncertainty_cache_{0.0f};
};

} // namespace tso
