#pragma once

#include "model.hpp"
#include "dataset.hpp"
#include "calibration.hpp"
#include "serializer.hpp"
#include "optimizer.hpp"
#include "loss.hpp"
#include "text_tokenizer.hpp"
#include <array>
#include <string>
#include <string_view>
#include <filesystem>
#include <chrono>
#include <format>
#include <stdexcept>
#include <unordered_map>

namespace tso {

inline std::vector<LayerConfig> default_classifier_topology() {
    return {
        {24, 32, Activation::GELU},
        {32, 16, Activation::GELU}
    };
}

inline std::vector<LayerConfig> default_text_classifier_topology() {
    return {
        {24, 48, Activation::GELU},
        {48, 24, Activation::GELU}
    };
}

// High-level input observation
struct Observation {
    StructuredState state;
    PresenceMask mask{
        .declared = true,
        .registered = true,
        .runtime = true,
        .witness = true,
        .freshness = true,
        .health = true
    };

    [[nodiscard]] Vector encode() const {
        return state.encode_with_mask(mask);
    }
};

// Rich typed classification result
struct ClassificationResult {
    // 1. Primary Typed Judgments
    Choice choice{Choice::Unknown};
    Noul locus{Noul::None};

    // 2. Continuous Scores & Epistemic Uncertainty
    Scalar score{0.0f};             // Viabilidade contínua estimada \hat{\mu}
    Scalar uncertainty{0.0f};       // Variância heteroscedástica estimada \hat{\sigma}^2

    // 3. Complete Calibrated Probability Distributions
    std::array<Scalar, 4> choice_probabilities{0.25f, 0.25f, 0.25f, 0.25f};
    std::array<Scalar, 7> locus_probabilities{0.0f};

    // 4. Calibration & Epistemic Diagnostics
    Scalar confidence{0.0f};        // Max P(choice)
    Scalar entropy{0.0f};           // Entropia de Shannon normalizada H(P) \in [0, 1]
    Scalar free_energy{0.0f};       // Helmholtz Free Energy E(x; T)

    // 5. Selective Prediction & Decision Routing
    bool abstained{false};          // True se for ruído alienígena / OOD
    std::string decision_status;    // Sumário textual (ex: "CONFIDENT_NOMINAL", "ABSTAINED_OOD")

    // 6. Provenance & Performance
    ModelIdentity model;
    double latency_nanoseconds{0.0};
};

struct ClassifierTrainingConfig {
    std::size_t epochs{80};
    std::size_t batch_size{16};
    Scalar learning_rate{0.008f};
    Scalar weight_decay{0.0005f};
    Scalar ood_percentile{0.98f};
    std::uint64_t seed{42};
};

struct TextClassifierTrainingConfig {
    std::size_t epochs{90};
    std::size_t batch_size{16};
    Scalar learning_rate{0.006f};
    Scalar weight_decay{0.0005f};
    Scalar ood_percentile{0.98f};
    std::uint64_t seed{101};
};

class SystemOneClassifier {
public:
    SystemOneClassifier()
        : model_(create_default_model_()) {}

    SystemOneClassifier(MultiHeadMLP model, TemperatureScaler scaler, SelectivePredictor predictor, ModelIdentity identity)
        : model_(std::move(model)), scaler_(std::move(scaler)), predictor_(std::move(predictor)), identity_(std::move(identity)) {}

    // Factory method: Trains and calibrates end-to-end on canonical dataset
    static SystemOneClassifier train_and_calibrate(
        const DatasetSplit& split,
        const ClassifierTrainingConfig& cfg = {}
    ) {
        Random rng(cfg.seed);
        MultiHeadMLP model(default_classifier_topology(), 16, 4, 7, rng);

        AdamWConfig opt_cfg{
            .lr = cfg.learning_rate,
            .beta1 = 0.9f,
            .beta2 = 0.999f,
            .eps = 1e-8f,
            .weight_decay = cfg.weight_decay
        };
        AdamW optimizer(opt_cfg);

        auto train_data = split.train;

        for (std::size_t epoch = 1; epoch <= cfg.epochs; ++epoch) {
            rng.shuffle(train_data);

            for (std::size_t i = 0; i < train_data.size(); i += cfg.batch_size) {
                const std::size_t cur_batch = std::min(cfg.batch_size, train_data.size() - i);
                model.zero_grad();

                for (std::size_t b = 0; b < cur_batch; ++b) {
                    const auto& sample = train_data[i + b];
                    auto out = model.forward(sample.x);

                    auto [c_loss, d_c] = CrossEntropyLoss::compute_from_index(out.choice_probs, sample.y);
                    auto [n_loss, d_n] = CrossEntropyLoss::compute_from_index(out.noul_probs, sample.noul_target);
                    auto [s_loss, d_s] = MSELoss::compute_scalar(out.score, sample.score_target);

                    const Scalar scale = 1.0f / static_cast<Scalar>(cur_batch);
                    for (auto& g : d_c) g *= scale;
                    for (auto& g : d_n) g *= scale;
                    for (auto& g : d_s) g = g * 2.0f * scale;

                    model.backward(d_c, d_n, d_s, Vector{0.0f});
                }

                model.update(optimizer);
            }
        }

        // Post-Hoc Temperature Calibration
        std::vector<Vector> val_logits;
        std::vector<std::size_t> val_targets;
        val_logits.reserve(split.val.size());
        val_targets.reserve(split.val.size());

        for (const auto& sample : split.val) {
            model.forward(sample.x);
            val_logits.push_back(model.choice_logits());
            val_targets.push_back(sample.y);
        }

        TemperatureScaler scaler;
        scaler.fit(val_logits, val_targets, 100, 0.05f);

        // OOD Threshold Calibration
        SelectivePredictor predictor;
        predictor.fit_threshold(val_logits, cfg.ood_percentile);

        ModelIdentity identity{
            .model_name = "TinySystemOne-ProductionClassifier",
            .version = "1.0.0",
            .trained_timestamp = static_cast<std::uint64_t>(std::chrono::system_clock::now().time_since_epoch().count()),
            .parameter_count = model.num_params(),
            .calibrated_temperature = scaler.temperature,
            .ood_energy_threshold = predictor.energy_threshold
        };

        return SystemOneClassifier(std::move(model), std::move(scaler), std::move(predictor), std::move(identity));
    }

    // High-Level Inference from Structured Observation
    [[nodiscard]] ClassificationResult classify(const Observation& obs) const {
        return classify_raw(obs.encode());
    }

    [[nodiscard]] ClassificationResult classify(const StructuredState& state, const PresenceMask& mask = {}) const {
        Observation obs{.state = state, .mask = mask};
        return classify(obs);
    }

    // Direct Inference from 24D Encoded Vector
    [[nodiscard]] ClassificationResult classify_raw(const Vector& x_24d) const {
        const auto t_start = std::chrono::steady_clock::now();

        auto out = const_cast<MultiHeadMLP&>(model_).forward(x_24d);
        const auto& logits = model_.choice_logits();

        // Temperature-scaled probabilities for Choice
        Vector calibrated_choice_probs = scaler_.calibrate(logits);
        const auto& noul_probs = out.noul_probs;

        // Energy OOD and Selective Decision
        auto selective_decision = predictor_.evaluate(logits, calibrated_choice_probs);

        const auto t_end = std::chrono::steady_clock::now();
        const double latency_ns = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count()
        );

        ClassificationResult res;
        res.model = identity_;
        res.latency_nanoseconds = latency_ns;
        res.free_energy = selective_decision.energy;
        res.abstained = selective_decision.abstained;

        if (res.abstained) {
            res.choice = Choice::Unknown;
            res.locus = Noul::None;
            res.score = 0.0f;
            res.uncertainty = 1.0f;
            res.choice_probabilities = {0.25f, 0.25f, 0.25f, 0.25f};
            res.locus_probabilities.fill(1.0f / 7.0f);
            res.confidence = 0.25f;
            res.entropy = 1.0f;
            res.decision_status = "ABSTAINED_OOD";
            return res;
        }

        // Fill In-Distribution Judgments
        for (std::size_t i = 0; i < 4 && i < calibrated_choice_probs.size(); ++i) {
            res.choice_probabilities[i] = calibrated_choice_probs[i];
        }

        for (std::size_t i = 0; i < 7 && i < noul_probs.size(); ++i) {
            res.locus_probabilities[i] = noul_probs[i];
        }

        auto max_c_it = std::max_element(calibrated_choice_probs.begin(), calibrated_choice_probs.end());
        const std::size_t choice_idx = static_cast<std::size_t>(std::distance(calibrated_choice_probs.begin(), max_c_it));
        res.choice = static_cast<Choice>(choice_idx);
        res.confidence = *max_c_it;
        res.entropy = Calibration::normalized_entropy(calibrated_choice_probs);

        auto max_n_it = std::max_element(noul_probs.begin(), noul_probs.end());
        const std::size_t locus_idx = static_cast<std::size_t>(std::distance(noul_probs.begin(), max_n_it));
        res.locus = static_cast<Noul>(locus_idx);

        res.score = out.score;
        res.uncertainty = out.uncertainty;

        // Build human-friendly decision status
        if (res.choice == Choice::Nominal && res.confidence >= 0.90f) {
            res.decision_status = "CONFIDENT_NOMINAL";
        } else if (res.choice == Choice::Inconsistent) {
            res.decision_status = "CONTRADICTION_RESOLVED";
        } else if (res.choice == Choice::Degraded) {
            res.decision_status = "DEGRADED_LOCALIZED";
        } else if (res.choice == Choice::Unknown) {
            res.decision_status = "MISSING_EVIDENCE_UNKNOWN";
        } else {
            res.decision_status = "LOW_CONFIDENCE";
        }

        return res;
    }

    // Persistence
    void save(const std::filesystem::path& path) const {
        Serializer::save(path, model_, scaler_, predictor_, identity_);
    }

    static SystemOneClassifier load(const std::filesystem::path& path) {
        Random dummy_rng(0);
        MultiHeadMLP model(default_classifier_topology(), 16, 4, 7, dummy_rng);
        TemperatureScaler scaler;
        SelectivePredictor predictor;
        ModelIdentity identity;

        Serializer::load(path, model, scaler, predictor, identity);
        return SystemOneClassifier(std::move(model), std::move(scaler), std::move(predictor), std::move(identity));
    }

    [[nodiscard]] const ModelIdentity& identity() const { return identity_; }
    [[nodiscard]] const MultiHeadMLP& raw_model() const { return model_; }
    [[nodiscard]] const TemperatureScaler& scaler() const { return scaler_; }
    [[nodiscard]] const SelectivePredictor& predictor() const { return predictor_; }

private:
    static MultiHeadMLP create_default_model_() {
        Random rng(42);
        return MultiHeadMLP(default_classifier_topology(), 16, 4, 7, rng);
    }

    MultiHeadMLP model_;
    TemperatureScaler scaler_;
    SelectivePredictor predictor_;
    ModelIdentity identity_;
};

/**
 * @brief Production Text-Transformer Classifier for Typed Probabilistic Judgment over Textual States (v1.0.0 / EXP-010).
 *
 * Direct end-to-end inference from semi-structured or natural text descriptions
 * into typed choices, diagnostic loci, continuous scores, calibrated probabilities, and OOD detection.
 */
class SystemOneTextClassifier {
public:
    SystemOneTextClassifier()
        : tokenizer_(), model_(create_default_text_model_()) {}

    SystemOneTextClassifier(
        TextTokenizer tokenizer,
        AttentionMultiHeadMLP model,
        TemperatureScaler scaler,
        SelectivePredictor predictor,
        ModelIdentity identity
    ) : tokenizer_(std::move(tokenizer)),
        model_(std::move(model)),
        scaler_(std::move(scaler)),
        predictor_(std::move(predictor)),
        identity_(std::move(identity)) {}

    static SystemOneTextClassifier train_and_calibrate(
        const DatasetSplit& split,
        const TextClassifierTrainingConfig& cfg = {}
    ) {
        Random rng(cfg.seed);
        TextTokenizer tokenizer;

        constexpr std::size_t kEmbeddingDim = 24;
        constexpr std::size_t kHeads = 3;
        constexpr std::size_t kFFDim = 48;

        AttentionMultiHeadMLP model(
            tokenizer.vocab_size(),
            TextTokenizer::kMaxSequenceLength,
            kEmbeddingDim,
            kHeads,
            kFFDim,
            default_text_classifier_topology(),
            24, 4, 7, rng, static_cast<uint32_t>(cfg.seed)
        );

        AdamWConfig opt_cfg{
            .lr = cfg.learning_rate,
            .beta1 = 0.9f,
            .beta2 = 0.999f,
            .eps = 1e-8f,
            .weight_decay = cfg.weight_decay
        };
        AdamW optimizer(opt_cfg);

        // Pre-build state universe mapping for training samples
        auto canonical_universe = DatasetGenerator::generate_canonical_universe();
        std::unordered_map<std::size_t, StructuredState> state_lookup;
        for (const auto& s : canonical_universe) {
            state_lookup[s.state_id()] = s;
        }

        struct TextTrainItem {
            std::vector<TokenId> tokens;
            std::size_t y;
            std::size_t noul_target;
            Scalar score_target;
            Scalar uncertainty_target;
        };

        std::vector<TextTrainItem> train_items;
        train_items.reserve(split.train.size() * 4);

        // Augment training data across 4 syntax styles
        for (const auto& sample : split.train) {
            auto it = state_lookup.find(sample.state_id);
            if (it == state_lookup.end()) continue;
            const auto& st = it->second;

            for (std::size_t s = 0; s < 4; ++s) {
                auto style = static_cast<TextualStyle>(s);
                std::string text = TextualStateGenerator::generate_text(st, style);
                train_items.push_back({
                    .tokens = tokenizer.tokenize(text),
                    .y = sample.y,
                    .noul_target = sample.noul_target,
                    .score_target = sample.score_target,
                    .uncertainty_target = sample.uncertainty_target
                });
            }
        }

        // Training loop
        for (std::size_t epoch = 1; epoch <= cfg.epochs; ++epoch) {
            rng.shuffle(train_items);

            for (std::size_t i = 0; i < train_items.size(); i += cfg.batch_size) {
                const std::size_t cur_batch = std::min(cfg.batch_size, train_items.size() - i);
                model.zero_grad();

                for (std::size_t b = 0; b < cur_batch; ++b) {
                    const auto& item = train_items[i + b];
                    auto out = model.forward(item.tokens);

                    auto [c_loss, d_c] = CrossEntropyLoss::compute_from_index(out.choice_probs, item.y);
                    auto [n_loss, d_n] = CrossEntropyLoss::compute_from_index(out.noul_probs, item.noul_target);
                    auto [s_loss, d_s] = MSELoss::compute_scalar(out.score, item.score_target);

                    const Scalar scale = 1.0f / static_cast<Scalar>(cur_batch);
                    for (auto& g : d_c) g *= scale;
                    for (auto& g : d_n) g *= scale;
                    for (auto& g : d_s) g = g * 2.0f * scale;

                    model.backward(d_c, d_n, d_s, Vector{0.0f});
                }

                model.update(optimizer);
            }
        }

        // Calibration on Validation split
        std::vector<Vector> val_logits;
        std::vector<std::size_t> val_targets;
        val_logits.reserve(split.val.size() * 4);
        val_targets.reserve(split.val.size() * 4);

        for (const auto& sample : split.val) {
            auto it = state_lookup.find(sample.state_id);
            if (it == state_lookup.end()) continue;
            const auto& st = it->second;

            for (std::size_t s = 0; s < 4; ++s) {
                auto style = static_cast<TextualStyle>(s);
                std::string text = TextualStateGenerator::generate_text(st, style);
                auto tokens = tokenizer.tokenize(text);
                model.forward(tokens);
                val_logits.push_back(model.choice_logits());
                val_targets.push_back(sample.y);
            }
        }

        TemperatureScaler scaler;
        scaler.fit(val_logits, val_targets, 100, 0.05f);

        SelectivePredictor predictor;
        predictor.fit_threshold(val_logits, cfg.ood_percentile);

        ModelIdentity identity{
            .model_name = "TinySystemOne-TextTransformerClassifier",
            .version = "1.0.0",
            .trained_timestamp = static_cast<std::uint64_t>(std::chrono::system_clock::now().time_since_epoch().count()),
            .parameter_count = model.num_params(),
            .calibrated_temperature = scaler.temperature,
            .ood_energy_threshold = predictor.energy_threshold
        };

        return SystemOneTextClassifier(std::move(tokenizer), std::move(model), std::move(scaler), std::move(predictor), std::move(identity));
    }

    [[nodiscard]] ClassificationResult classify(std::string_view text) const {
        const auto t_start = std::chrono::steady_clock::now();
        auto tokens = tokenizer_.tokenize(text);
        return classify_tokens_internal(tokens, t_start);
    }

    [[nodiscard]] ClassificationResult classify_tokens(const std::vector<TokenId>& tokens) const {
        const auto t_start = std::chrono::steady_clock::now();
        return classify_tokens_internal(tokens, t_start);
    }

    void save(const std::filesystem::path& path) const {
        Serializer::save_text_model(path, model_, scaler_, predictor_, identity_);
    }

    static SystemOneTextClassifier load(const std::filesystem::path& path) {
        Random dummy_rng(0);
        TextTokenizer tokenizer;
        constexpr std::size_t kEmbeddingDim = 24;
        constexpr std::size_t kHeads = 3;
        constexpr std::size_t kFFDim = 48;

        AttentionMultiHeadMLP model(
            tokenizer.vocab_size(),
            TextTokenizer::kMaxSequenceLength,
            kEmbeddingDim,
            kHeads,
            kFFDim,
            default_text_classifier_topology(),
            24, 4, 7, dummy_rng, 0
        );

        TemperatureScaler scaler;
        SelectivePredictor predictor;
        ModelIdentity identity;

        Serializer::load_text_model(path, model, scaler, predictor, identity);
        return SystemOneTextClassifier(std::move(tokenizer), std::move(model), std::move(scaler), std::move(predictor), std::move(identity));
    }

    [[nodiscard]] const ModelIdentity& identity() const { return identity_; }
    [[nodiscard]] const AttentionMultiHeadMLP& raw_model() const { return model_; }
    [[nodiscard]] const TemperatureScaler& scaler() const { return scaler_; }
    [[nodiscard]] const SelectivePredictor& predictor() const { return predictor_; }
    [[nodiscard]] const TextTokenizer& tokenizer() const { return tokenizer_; }

private:
    static AttentionMultiHeadMLP create_default_text_model_() {
        Random rng(42);
        TextTokenizer tok;
        return AttentionMultiHeadMLP(
            tok.vocab_size(),
            TextTokenizer::kMaxSequenceLength,
            24, 3, 48,
            default_text_classifier_topology(),
            24, 4, 7, rng, 42
        );
    }

    [[nodiscard]] ClassificationResult classify_tokens_internal(
        const std::vector<TokenId>& tokens,
        std::chrono::steady_clock::time_point t_start
    ) const {
        auto out = const_cast<AttentionMultiHeadMLP&>(model_).forward(tokens);
        const auto& logits = model_.choice_logits();

        Vector calibrated_choice_probs = scaler_.calibrate(logits);
        const auto& noul_probs = out.noul_probs;

        auto selective_decision = predictor_.evaluate(logits, calibrated_choice_probs);

        const auto t_end = std::chrono::steady_clock::now();
        const double latency_ns = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count()
        );

        ClassificationResult res;
        res.model = identity_;
        res.latency_nanoseconds = latency_ns;
        res.free_energy = selective_decision.energy;
        res.abstained = selective_decision.abstained;

        if (res.abstained) {
            res.choice = Choice::Unknown;
            res.locus = Noul::None;
            res.score = 0.0f;
            res.uncertainty = 1.0f;
            res.choice_probabilities = {0.25f, 0.25f, 0.25f, 0.25f};
            res.locus_probabilities.fill(1.0f / 7.0f);
            res.confidence = 0.25f;
            res.entropy = 1.0f;
            res.decision_status = "ABSTAINED_OOD";
            return res;
        }

        for (std::size_t i = 0; i < 4 && i < calibrated_choice_probs.size(); ++i) {
            res.choice_probabilities[i] = calibrated_choice_probs[i];
        }

        for (std::size_t i = 0; i < 7 && i < noul_probs.size(); ++i) {
            res.locus_probabilities[i] = noul_probs[i];
        }

        auto max_c_it = std::max_element(calibrated_choice_probs.begin(), calibrated_choice_probs.end());
        const std::size_t choice_idx = static_cast<std::size_t>(std::distance(calibrated_choice_probs.begin(), max_c_it));
        res.choice = static_cast<Choice>(choice_idx);
        res.confidence = *max_c_it;
        res.entropy = Calibration::normalized_entropy(calibrated_choice_probs);

        auto max_n_it = std::max_element(noul_probs.begin(), noul_probs.end());
        const std::size_t locus_idx = static_cast<std::size_t>(std::distance(noul_probs.begin(), max_n_it));
        res.locus = static_cast<Noul>(locus_idx);

        res.score = out.score;
        res.uncertainty = out.uncertainty;

        if (res.choice == Choice::Nominal && res.confidence >= 0.90f) {
            res.decision_status = "CONFIDENT_NOMINAL";
        } else if (res.choice == Choice::Inconsistent) {
            res.decision_status = "CONTRADICTION_RESOLVED";
        } else if (res.choice == Choice::Degraded) {
            res.decision_status = "DEGRADED_LOCALIZED";
        } else if (res.choice == Choice::Unknown) {
            res.decision_status = "MISSING_EVIDENCE_UNKNOWN";
        } else {
            res.decision_status = "LOW_CONFIDENCE";
        }

        return res;
    }

    TextTokenizer tokenizer_;
    AttentionMultiHeadMLP model_;
    TemperatureScaler scaler_;
    SelectivePredictor predictor_;
    ModelIdentity identity_;
};

} // namespace tso
