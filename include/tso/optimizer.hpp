#pragma once

#include "tensor.hpp"
#include <vector>
#include <cmath>

namespace tso {

struct AdamWConfig {
    Scalar lr{0.005f};
    Scalar beta1{0.9f};
    Scalar beta2{0.999f};
    Scalar eps{1e-8f};
    Scalar weight_decay{0.01f};
};

struct ParameterState {
    Vector m;
    Vector v;
};

class AdamW {
public:
    explicit AdamW(const AdamWConfig& config = {}) : config_(config) {}

    void step(Matrix& W, const Matrix& dW, ParameterState& state) {
        ensure_state(state, W.data.size());
        step_ = std::max(step_ + 1, std::size_t(1));
        
        const Scalar beta1_t = std::pow(config_.beta1, static_cast<Scalar>(step_));
        const Scalar beta2_t = std::pow(config_.beta2, static_cast<Scalar>(step_));
        const Scalar alpha_t = config_.lr * std::sqrt(1.0f - beta2_t) / (1.0f - beta1_t);

        for (std::size_t i = 0; i < W.data.size(); ++i) {
            const Scalar g = dW.data[i];
            
            // Decoupled weight decay
            W.data[i] -= config_.lr * config_.weight_decay * W.data[i];

            // Moment updates
            state.m[i] = config_.beta1 * state.m[i] + (1.0f - config_.beta1) * g;
            state.v[i] = config_.beta2 * state.v[i] + (1.0f - config_.beta2) * g * g;

            // Parameter update
            W.data[i] -= alpha_t * state.m[i] / (std::sqrt(state.v[i]) + config_.eps);
        }
    }

    void step_vector(Vector& b, const Vector& db, ParameterState& state) {
        ensure_state(state, b.size());
        // Biases typically do not have weight decay applied
        const Scalar beta1_t = std::pow(config_.beta1, static_cast<Scalar>(step_));
        const Scalar beta2_t = std::pow(config_.beta2, static_cast<Scalar>(step_));
        const Scalar alpha_t = config_.lr * std::sqrt(1.0f - beta2_t) / (1.0f - beta1_t);

        for (std::size_t i = 0; i < b.size(); ++i) {
            const Scalar g = db[i];
            state.m[i] = config_.beta1 * state.m[i] + (1.0f - config_.beta1) * g;
            state.v[i] = config_.beta2 * state.v[i] + (1.0f - config_.beta2) * g * g;
            b[i] -= alpha_t * state.m[i] / (std::sqrt(state.v[i]) + config_.eps);
        }
    }

    void reset() {
        step_ = 0;
    }

    void set_lr(Scalar lr) {
        config_.lr = lr;
    }

    [[nodiscard]] Scalar lr() const {
        return config_.lr;
    }

private:
    void ensure_state(ParameterState& state, std::size_t size) {
        if (state.m.size() != size) {
            state.m.assign(size, 0.0f);
            state.v.assign(size, 0.0f);
        }
    }

    AdamWConfig config_;
    std::size_t step_{0};
};

} // namespace tso
