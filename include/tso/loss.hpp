#pragma once

#include "tensor.hpp"
#include <cmath>
#include <cstddef>
#include <algorithm>

namespace tso {

struct LossResult {
    Scalar loss{0.0f};
    Vector dlogits; // Gradient with respect to pre-softmax logits
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

} // namespace tso
