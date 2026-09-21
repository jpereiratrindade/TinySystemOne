#include <tso/tensor.hpp>
#include <tso/random.hpp>
#include <tso/loss.hpp>
#include <tso/model.hpp>
#include <tso/optimizer.hpp>
#include <tso/calibration.hpp>
#include <tso/dataset.hpp>

#include <iostream>
#include <cstdlib>
#include <format>
#include <cmath>

#define ACCEPTANCE_ASSERT(expr, msg) \
    do { \
        if (!(expr)) { \
            std::cerr << "\033[1;31m[ACCEPTANCE FAILED]\033[0m " << (msg) << " (" #expr ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(1); \
        } \
    } while (0)

void test_disjoint_partitions() {
    std::cout << "[ACCEPTANCE] Verifying zero state leakage across train/val/test...\n";
    auto split = tso::DatasetGenerator::generate_canonical_split(42);
    
    // Verify disjoint sets
    tso::DatasetGenerator::verify_disjoint_partitions(split);

    ACCEPTANCE_ASSERT(!split.train.empty(), "Train split is empty");
    ACCEPTANCE_ASSERT(!split.val.empty(), "Val split is empty");
    ACCEPTANCE_ASSERT(!split.test.empty(), "Test split is empty");
    ACCEPTANCE_ASSERT(!split.ood.empty(), "OOD split is empty");

    std::cout << std::format("  ✓ Disjoint partitions confirmed! (Train: {} unique states, Val: {} unique states, Test: {} unique states)\n",
                             split.train.size(), split.val.size(), split.test.size());
}

void test_generalization_on_unseen_states() {
    std::cout << "[ACCEPTANCE] Training and evaluating generalization on genuinely unseen state configurations...\n";
    constexpr std::uint64_t kSeed = 101;
    tso::Random rng(kSeed);

    std::vector<tso::LayerConfig> trunk_topo = {
        {24, 32, tso::Activation::GELU},
        {32, 16, tso::Activation::GELU}
    };

    tso::MultiHeadMLP model(trunk_topo, 16, 4, 7, rng);
    auto split = tso::DatasetGenerator::generate_canonical_split(kSeed);

    tso::AdamWConfig opt_cfg{
        .lr = 0.008f,
        .beta1 = 0.9f,
        .beta2 = 0.999f,
        .eps = 1e-8f,
        .weight_decay = 0.0005f
    };
    tso::AdamW optimizer(opt_cfg);

    constexpr std::size_t kEpochs = 80;
    constexpr std::size_t kBatchSize = 16;

    for (std::size_t epoch = 1; epoch <= kEpochs; ++epoch) {
        rng.shuffle(split.train);
        for (std::size_t i = 0; i < split.train.size(); i += kBatchSize) {
            const std::size_t current_batch_size = std::min(kBatchSize, split.train.size() - i);
            model.zero_grad();

            for (std::size_t b = 0; b < current_batch_size; ++b) {
                const auto& sample = split.train[i + b];
                auto out = model.forward(sample.x);

                auto [c_loss, d_c] = tso::CrossEntropyLoss::compute_from_index(out.choice_probs, sample.y);
                auto [n_loss, d_n] = tso::CrossEntropyLoss::compute_from_index(out.noul_probs, sample.noul_target);
                auto [s_loss, d_s] = tso::MSELoss::compute_scalar(out.score, sample.score_target);

                const float scale = 1.0f / static_cast<float>(current_batch_size);
                for (auto& g : d_c) g *= scale;
                for (auto& g : d_n) g *= scale;
                for (auto& g : d_s) g = g * 2.0f * scale;

                model.backward(d_c, d_n, d_s, tso::Vector{0.0f});
            }

            model.update(optimizer);
        }
    }

    // Evaluate on 100% UNSEEN unique state configurations in test split
    std::vector<tso::Vector> test_c_probs;
    std::vector<tso::Vector> test_n_probs;
    std::vector<std::size_t> test_c_targets;
    std::vector<std::size_t> test_n_targets;

    for (const auto& sample : split.test) {
        auto out = model.forward(sample.x);
        test_c_probs.push_back(out.choice_probs);
        test_n_probs.push_back(out.noul_probs);
        test_c_targets.push_back(sample.y);
        test_n_targets.push_back(sample.noul_target);
    }

    auto c_report = tso::Calibration::evaluate(test_c_probs, test_c_targets, 10);
    auto n_report = tso::Calibration::evaluate(test_n_probs, test_n_targets, 10);

    std::cout << std::format("  • Choice Accuracy on Unseen States: {:.2f}% (ECE: {:.4f})\n", c_report.accuracy * 100.0f, c_report.ece);
    std::cout << std::format("  • Noul Accuracy on Unseen States:   {:.2f}% (ECE: {:.4f})\n", n_report.accuracy * 100.0f, n_report.ece);

    ACCEPTANCE_ASSERT(c_report.accuracy >= 0.88f, "Choice generalization on unseen states must be >= 88%");
    ACCEPTANCE_ASSERT(n_report.accuracy >= 0.85f, "Noul generalization on unseen states must be >= 85%");
    ACCEPTANCE_ASSERT(c_report.ece <= 0.08f, "Choice ECE must be <= 0.08");

    // 1. Direct Factual Contradiction in valid state
    tso::StructuredState direct_contradiction{
        .declared = false,
        .registered = true,
        .runtime = tso::RuntimeState::Running,
        .witness = tso::WitnessState::Valid,
        .freshness = tso::FreshnessState::Fresh,
        .health = tso::HealthState::Healthy
    };
    auto out_direct = model.forward(direct_contradiction.encode());
    float p_inc_direct = out_direct.choice_probs[static_cast<std::size_t>(tso::Choice::Inconsistent)];
    std::cout << std::format("  • Direct Factual Contradiction P(INCONSISTENT): {:.4f}\n", p_inc_direct);
    ACCEPTANCE_ASSERT(p_inc_direct >= 0.85f, "Direct contradiction must produce high P(INCONSISTENT) >= 0.85");

    // 2. Test OOD Taxonomy
    for (const auto& ood : split.ood) {
        auto out = model.forward(ood.x);
        if (ood.category == tso::OodCategory::NoSignal) {
            float norm_ent = tso::Calibration::normalized_entropy(out.choice_probs);
            std::cout << std::format("  • NO_SIGNAL Normalized Entropy: {:.4f}\n", norm_ent);
            ACCEPTANCE_ASSERT(norm_ent >= 0.60f, "NO_SIGNAL must produce high normalized entropy (>= 0.60)");
        } else if (ood.category == tso::OodCategory::UniformDispersion) {
            float norm_ent = tso::Calibration::normalized_entropy(out.choice_probs);
            std::cout << std::format("  • UNIFORM_DISPERSION Normalized Entropy: {:.4f}\n", norm_ent);
            ACCEPTANCE_ASSERT(norm_ent >= 0.50f, "UNIFORM_DISPERSION must produce high entropy");
        }
    }

    std::cout << "  ✓ Scientific acceptance criteria passed on unseen configurations and OOD taxonomy!\n";
}

int main() {
    std::cout << "=== TinySystemOne Rigorous Scientific Acceptance Tests ===\n";
    test_disjoint_partitions();
    test_generalization_on_unseen_states();
    std::cout << "\033[1;32mAll acceptance criteria satisfied with zero data leakage!\033[0m\n";
    return 0;
}
