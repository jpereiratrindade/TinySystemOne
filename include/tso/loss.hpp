#pragma once

#include "tensor.hpp"
#include <cmath>
#include <cstddef>
#include <algorithm>

namespace tso {

struct LossResult {
    Scalar loss{0.0f};
    Vector dlogits; // Gradient with respect to pre-activation / pre-softmax logits
};

class CrossEntropyLoss {
public:
    static constexpr Scalar kEpsilon = 1e-12f;

    // Target given as one-hot or probability distribution
    static LossResult compute(const Vector& probs, const Vector& target) {
        Scalar loss = 0.0f;
        Vector dlogits(probs.size());
        for (std::size_t i = 0; i < probs.size(); ++i) {
            const Scalar p = std::clamp(probs[i], kEpsilon, 1.0f - kEpsilon);
            loss -= target[i] * std::log(p);
            dlogits[i] = probs[i] - target[i];
        }
        return {loss, std::move(dlogits)};
    }

    // Target given as class index
    static LossResult compute_from_index(const Vector& probs, std::size_t target_idx) {
        Scalar loss = 0.0f;
        Vector dlogits(probs.size());
        for (std::size_t i = 0; i < probs.size(); ++i) {
            const Scalar target_i = (i == target_idx) ? 1.0f : 0.0f;
            const Scalar p = std::clamp(probs[i], kEpsilon, 1.0f - kEpsilon);
            loss -= target_i * std::log(p);
            dlogits[i] = probs[i] - target_i;
        }
        return {loss, std::move(dlogits)};
    }
};

class MSELoss {
public:
    static LossResult compute(const Vector& pred, const Vector& target) {
        Scalar loss = 0.0f;
        Vector dpred(pred.size());
        for (std::size_t i = 0; i < pred.size(); ++i) {
            const Scalar diff = pred[i] - target[i];
            loss += diff * diff;
            dpred[i] = 2.0f * diff;
        }
        return {loss, std::move(dpred)};
    }

    static LossResult compute_scalar(Scalar pred, Scalar target) {
        const Scalar diff = pred - target;
        return {diff * diff, Vector{2.0f * diff}};
    }
};

struct NLLResult {
    Scalar loss{0.0f};
    Vector d_mu;
    Vector d_var;
};

class GaussianNLLLoss {
public:
    static constexpr Scalar kEps = 1e-6f;

    // L(mu, var, y) = 0.5 * (mu - y)^2 / var + 0.5 * ln(var)
    // dL/dmu = (mu - y) / var
    // dL/dvar = -0.5 * (mu - y)^2 / var^2 + 0.5 / var
    static NLLResult compute(Scalar mu, Scalar var, Scalar target) {
        const Scalar clamped_var = std::max(var, kEps);
        const Scalar diff = mu - target;
        const Scalar diff_sq = diff * diff;
        
        const Scalar loss = 0.5f * (diff_sq / clamped_var) + 0.5f * std::log(clamped_var);
        const Scalar d_mu = diff / clamped_var;
        const Scalar d_var = -0.5f * (diff_sq / (clamped_var * clamped_var)) + 0.5f / clamped_var;

        return {loss, Vector{d_mu}, Vector{d_var}};
    }
};

struct InvarianceLossResult {
    Scalar loss{0.0f};
    Vector grad_h1;
    Vector grad_h2;
};

class InvarianceLoss {
public:
    // L_inv = 0.5 * ||h1 - h2||^2
    // dL/dh1 = (h1 - h2), dL/dh2 = (h2 - h1)
    static InvarianceLossResult compute(const Vector& h1, const Vector& h2) {
        Scalar loss = 0.0f;
        Vector grad_h1(h1.size());
        Vector grad_h2(h2.size());

        for (std::size_t i = 0; i < h1.size(); ++i) {
            const Scalar diff = h1[i] - h2[i];
            loss += 0.5f * diff * diff;
            grad_h1[i] = diff;
            grad_h2[i] = -diff;
        }

        return {loss, std::move(grad_h1), std::move(grad_h2)};
    }
};

} // namespace tso

