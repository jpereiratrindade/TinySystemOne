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

} // namespace tso
