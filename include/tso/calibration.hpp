#pragma once

#include "tensor.hpp"
#include <vector>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace tso {

struct BinMetrics {
    std::size_t bin_index{0};
    Scalar conf_lower{0.0f};
    Scalar conf_upper{0.0f};
    std::size_t count{0};
    Scalar avg_confidence{0.0f};
    Scalar accuracy{0.0f};
    Scalar calibration_gap{0.0f};
};

struct CalibrationReport {
    Scalar ece{0.0f};          // Expected Calibration Error
    Scalar mce{0.0f};          // Maximum Calibration Error
    Scalar brier_score{0.0f};  // Mean Brier Score
    Scalar accuracy{0.0f};     // Global accuracy
    Scalar avg_entropy{0.0f};  // Mean Shannon Entropy
    Scalar avg_confidence{0.0f};
    std::vector<BinMetrics> bins;
};

class Calibration {
public:
    static constexpr Scalar kEps = 1e-12f;

    // Shannon Entropy in nats: H(p) = - sum p_i ln(p_i)
    static Scalar shannon_entropy(const Vector& probs) {
        Scalar h = 0.0f;
        for (Scalar p : probs) {
            if (p > kEps) {
                h -= p * std::log(p);
            }
        }
        return h;
    }

    // Normalized Shannon Entropy in [0, 1]: H(p) / ln(K)
    static Scalar normalized_entropy(const Vector& probs) {
        if (probs.size() <= 1) return 0.0f;
        const Scalar h = shannon_entropy(probs);
        const Scalar max_h = std::log(static_cast<Scalar>(probs.size()));
        return h / max_h;
    }

    // Single-sample Brier score: sum_k (p_k - y_k)^2
    static Scalar sample_brier_score(const Vector& probs, std::size_t target_idx) {
        Scalar brier = 0.0f;
        for (std::size_t i = 0; i < probs.size(); ++i) {
            const Scalar target = (i == target_idx) ? 1.0f : 0.0f;
            const Scalar diff = probs[i] - target;
            brier += diff * diff;
        }
        return brier;
    }

    // Evaluate full calibration report across multiple predictions
    static CalibrationReport evaluate(
        const std::vector<Vector>& all_probs,
        const std::vector<std::size_t>& all_targets,
        std::size_t num_bins = 10
    ) {
        if (all_probs.size() != all_targets.size() || all_probs.empty()) {
            throw std::invalid_argument("Size mismatch or empty data in Calibration::evaluate");
        }

        const std::size_t N = all_probs.size();
        Scalar total_brier = 0.0f;
        Scalar total_entropy = 0.0f;
        Scalar total_conf = 0.0f;
        std::size_t total_correct = 0;

        struct SampleEval {
            Scalar confidence;
            std::size_t pred_class;
            std::size_t true_class;
            bool correct;
        };

        std::vector<SampleEval> samples;
        samples.reserve(N);

        for (std::size_t i = 0; i < N; ++i) {
            const auto& probs = all_probs[i];
            const std::size_t target = all_targets[i];

            auto max_it = std::max_element(probs.begin(), probs.end());
            const std::size_t pred_idx = static_cast<std::size_t>(std::distance(probs.begin(), max_it));
            const Scalar conf = *max_it;
            const bool correct = (pred_idx == target);

            if (correct) ++total_correct;
            total_brier += sample_brier_score(probs, target);
            total_entropy += shannon_entropy(probs);
            total_conf += conf;

            samples.push_back({conf, pred_idx, target, correct});
        }

        CalibrationReport report;
        report.accuracy = static_cast<Scalar>(total_correct) / static_cast<Scalar>(N);
        report.brier_score = total_brier / static_cast<Scalar>(N);
        report.avg_entropy = total_entropy / static_cast<Scalar>(N);
        report.avg_confidence = total_conf / static_cast<Scalar>(N);

        report.bins.resize(num_bins);
        const Scalar bin_size = 1.0f / static_cast<Scalar>(num_bins);

        std::vector<std::vector<SampleEval>> bin_samples(num_bins);
        for (const auto& sample : samples) {
            std::size_t b_idx = static_cast<std::size_t>(sample.confidence / bin_size);
            if (b_idx >= num_bins) b_idx = num_bins - 1;
            bin_samples[b_idx].push_back(sample);
        }

        Scalar ece = 0.0f;
        Scalar mce = 0.0f;

        for (std::size_t b = 0; b < num_bins; ++b) {
            BinMetrics& bm = report.bins[b];
            bm.bin_index = b;
            bm.conf_lower = static_cast<Scalar>(b) * bin_size;
            bm.conf_upper = static_cast<Scalar>(b + 1) * bin_size;
            bm.count = bin_samples[b].size();

            if (bm.count > 0) {
                Scalar sum_conf = 0.0f;
                std::size_t bin_correct = 0;
                for (const auto& s : bin_samples[b]) {
                    sum_conf += s.confidence;
                    if (s.correct) ++bin_correct;
                }
                bm.avg_confidence = sum_conf / static_cast<Scalar>(bm.count);
                bm.accuracy = static_cast<Scalar>(bin_correct) / static_cast<Scalar>(bm.count);
                bm.calibration_gap = std::abs(bm.accuracy - bm.avg_confidence);

                const Scalar weight = static_cast<Scalar>(bm.count) / static_cast<Scalar>(N);
                ece += weight * bm.calibration_gap;
                mce = std::max(mce, bm.calibration_gap);
            } else {
                bm.avg_confidence = 0.0f;
                bm.accuracy = 0.0f;
                bm.calibration_gap = 0.0f;
            }
        }

        report.ece = ece;
        report.mce = mce;
        return report;
    }
};

class TemperatureScaler {
public:
    Scalar temperature{1.0f};

    [[nodiscard]] Vector calibrate(const Vector& logits) const {
        if (logits.empty()) return {};
        Vector scaled(logits.size());
        const Scalar inv_t = 1.0f / std::max(0.01f, temperature);
        for (std::size_t i = 0; i < logits.size(); ++i) {
            scaled[i] = logits[i] * inv_t;
        }
        return softmax(scaled);
    }

    void fit(
        const std::vector<Vector>& all_logits,
        const std::vector<std::size_t>& all_targets,
        std::size_t max_iters = 100,
        Scalar lr = 0.05f
    ) {
        if (all_logits.empty() || all_logits.size() != all_targets.size()) return;

        temperature = 1.0f;
        const Scalar N = static_cast<Scalar>(all_logits.size());

        for (std::size_t iter = 0; iter < max_iters; ++iter) {
            Scalar grad_T = 0.0f;
            const Scalar inv_t = 1.0f / temperature;
            const Scalar inv_t2 = inv_t * inv_t;

            for (std::size_t sample_i = 0; sample_i < all_logits.size(); ++sample_i) {
                const auto& z = all_logits[sample_i];
                const std::size_t y = all_targets[sample_i];

                Vector scaled_z(z.size());
                for (std::size_t k = 0; k < z.size(); ++k) scaled_z[k] = z[k] * inv_t;
                Vector p = softmax(scaled_z);

                // dL/dT = (1/T^2) * sum_i (p_i - y_i) * z_i
                Scalar sample_grad = 0.0f;
                for (std::size_t k = 0; k < z.size(); ++k) {
                    const Scalar target_k = (k == y) ? 1.0f : 0.0f;
                    sample_grad += (p[k] - target_k) * z[k];
                }
                grad_T += inv_t2 * sample_grad;
            }

            const Scalar avg_grad = grad_T / N;
            temperature -= lr * avg_grad;
            temperature = std::clamp(temperature, 0.05f, 10.0f);

            if (std::abs(avg_grad) < 1e-5f) {
                break;
            }
        }
    }
};

} // namespace tso
