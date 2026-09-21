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

void test_generalization_and_ood_energy() {
    std::cout << "[ACCEPTANCE] Training and evaluating generalization, energy AUROC and selective prediction...\n";
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
    std::vector<float> id_test_energies;

    for (const auto& sample : split.test) {
        auto out = model.forward(sample.x);
        test_c_probs.push_back(out.choice_probs);
        test_n_probs.push_back(out.noul_probs);
        test_c_targets.push_back(sample.y);
        test_n_targets.push_back(sample.noul_target);
        id_test_energies.push_back(tso::Calibration::energy_score(model.choice_logits()));
    }

    auto c_report = tso::Calibration::evaluate(test_c_probs, test_c_targets, 10);
    auto n_report = tso::Calibration::evaluate(test_n_probs, test_n_targets, 10);

    std::cout << std::format("  • Choice Accuracy on Unseen States: {:.2f}% (ECE: {:.4f})\n", c_report.accuracy * 100.0f, c_report.ece);
    std::cout << std::format("  • Noul Accuracy on Unseen States:   {:.2f}% (ECE: {:.4f})\n", n_report.accuracy * 100.0f, n_report.ece);

    ACCEPTANCE_ASSERT(c_report.accuracy >= 0.88f, "Choice generalization on unseen states must be >= 88%");
    ACCEPTANCE_ASSERT(n_report.accuracy >= 0.85f, "Noul generalization on unseen states must be >= 85%");
    ACCEPTANCE_ASSERT(c_report.ece <= 0.08f, "Choice ECE must be <= 0.08");

    // Energy AUROC Check for Alien/Unstructured OOD Noise
    std::vector<float> noise_ood_energies;
    for (const auto& ood : split.ood) {
        if (ood.category == tso::OodCategory::NoSignal ||
            ood.category == tso::OodCategory::UniformDispersion ||
            ood.category == tso::OodCategory::CorruptedInput) {
            model.forward(ood.x);
            noise_ood_energies.push_back(tso::Calibration::energy_score(model.choice_logits()));
        }
    }

    float noise_auroc = tso::Calibration::compute_auroc(id_test_energies, noise_ood_energies, true);
    std::cout << std::format("  • Energy Alien-Noise OOD AUROC: {:.4f}\n", noise_auroc);
    ACCEPTANCE_ASSERT(noise_auroc >= 0.80f, "Energy score AUROC against alien noise must be >= 0.80");

    // Selective Predictor
    std::vector<tso::Vector> val_logits;
    for (const auto& sample : split.val) {
        model.forward(sample.x);
        val_logits.push_back(model.choice_logits());
    }
    tso::SelectivePredictor predictor;
    predictor.fit_threshold(val_logits, 0.98f);

    // ID test set acceptance rate must be >= 90%
    std::size_t id_accepted = 0;
    for (const auto& sample : split.test) {
        model.forward(sample.x);
        auto decision = predictor.evaluate(model.choice_logits(), model.choice_probs());
        if (!decision.abstained) ++id_accepted;
    }
    const float id_accept_rate = static_cast<float>(id_accepted) / static_cast<float>(split.test.size());
    std::cout << std::format("  • ID Test Set Acceptance Rate: {:.2f}%\n", id_accept_rate * 100.0f);
    ACCEPTANCE_ASSERT(id_accept_rate >= 0.90f, "ID Test set acceptance rate must be >= 90%");

    // Alien OOD (Zero signal) must be abstained
    model.forward(split.ood[0].x);
    auto no_signal_decision = predictor.evaluate(model.choice_logits(), model.choice_probs());
    ACCEPTANCE_ASSERT(no_signal_decision.abstained, "Alien zero-signal OOD input must be rejected/abstained");

    std::cout << "  ✓ Scientific acceptance criteria passed on unseen configurations, energy AUROC and selective prediction!\n";
}

void test_semantic_invariance_and_latent_manifolds() {
    std::cout << "[ACCEPTANCE] Testing Semantic Equivalence & Latent Manifold Invariance (EXP-006)...\n";
    constexpr std::uint64_t kSeed = 101;
    tso::Random rng(kSeed);

    std::vector<tso::LayerConfig> trunk_topology = {
        {24, 32, tso::Activation::GELU},
        {32, 16, tso::Activation::GELU}
    };
    tso::MultiHeadMLP model(trunk_topology, 16, 4, 7, rng);
    auto split = tso::DatasetGenerator::generate_canonical_split(kSeed);
    auto canonical_universe = tso::DatasetGenerator::generate_canonical_universe();

    tso::AdamW optimizer(tso::AdamWConfig{.lr = 0.008f, .beta1 = 0.9f, .beta2 = 0.999f, .eps = 1e-8f, .weight_decay = 0.0005f});

    // Train with multi-representation alignment
    for (std::size_t epoch = 1; epoch <= 80; ++epoch) {
        rng.shuffle(split.train);
        for (std::size_t i = 0; i < split.train.size(); i += 16) {
            const std::size_t cur_batch = std::min(std::size_t{16}, split.train.size() - i);
            model.zero_grad();
            for (std::size_t b = 0; b < cur_batch; ++b) {
                const auto& sample = split.train[i + b];
                auto out_a = model.forward(sample.x);
                tso::Vector h_a = model.extract_latent(sample.x);

                auto [c_loss, d_c] = tso::CrossEntropyLoss::compute_from_index(out_a.choice_probs, sample.y);
                auto [n_loss, d_n] = tso::CrossEntropyLoss::compute_from_index(out_a.noul_probs, sample.noul_target);
                auto [s_loss, d_s] = tso::MSELoss::compute_scalar(out_a.score, sample.score_target);

                const float scale = 1.0f / static_cast<float>(cur_batch);
                for (auto& g : d_c) g *= scale;
                for (auto& g : d_n) g *= scale;
                for (auto& g : d_s) g = g * 2.0f * scale;
                model.backward(d_c, d_n, d_s, tso::Vector{0.0f});

                // Invariant pair
                const auto& state = canonical_universe[sample.state_id];
                const auto form = static_cast<tso::EncodingForm>(1 + (b % 3));
                tso::Vector x_alt = state.encode_as(form);
                auto out_alt = model.forward(x_alt);
                tso::Vector h_alt = model.extract_latent(x_alt);

                auto [c_alt, d_c_alt] = tso::CrossEntropyLoss::compute_from_index(out_alt.choice_probs, sample.y);
                for (auto& g : d_c_alt) g *= (scale * 0.5f);
                model.backward(d_c_alt, tso::Vector(7, 0.0f), tso::Vector{0.0f}, tso::Vector{0.0f});

                auto inv_res = tso::InvarianceLoss::compute(h_a, h_alt);
                for (auto& g : inv_res.grad_h1) g *= (scale * 0.15f);
                for (auto& g : inv_res.grad_h2) g *= (scale * 0.15f);
                model.backward_trunk_from_latent(inv_res.grad_h1);
                model.backward_trunk_from_latent(inv_res.grad_h2);
            }
            model.update(optimizer);
        }
    }

    double sim_bipolar_sum = 0.0;
    double sim_ordinal_sum = 0.0;
    std::size_t matches = 0;
    std::size_t total = 0;

    for (const auto& st : canonical_universe) {
        tso::Vector x_a = st.encode_as(tso::EncodingForm::CanonicalOneHot);
        tso::Vector x_b = st.encode_as(tso::EncodingForm::NormalizedOrdinal);
        tso::Vector x_d = st.encode_as(tso::EncodingForm::BipolarDifferential);

        auto out_a = model.forward(x_a);
        tso::Vector h_a = model.extract_latent(x_a);

        auto out_b = model.forward(x_b);
        tso::Vector h_b = model.extract_latent(x_b);

        auto out_d = model.forward(x_d);
        tso::Vector h_d = model.extract_latent(x_d);

        sim_ordinal_sum += tso::Calibration::cosine_similarity(h_a, h_b);
        sim_bipolar_sum += tso::Calibration::cosine_similarity(h_a, h_d);

        auto get_idx = [](const tso::Vector& p) {
            return std::distance(p.begin(), std::max_element(p.begin(), p.end()));
        };
        if (get_idx(out_a.choice_probs) == get_idx(out_d.choice_probs)) ++matches;
        ++total;
    }

    const double mean_sim_ordinal = sim_ordinal_sum / static_cast<double>(total);
    const double mean_sim_bipolar = sim_bipolar_sum / static_cast<double>(total);
    const double agreement_bipolar = static_cast<double>(matches) / static_cast<double>(total);

    std::cout << std::format("  • Mean Latent Cosine Similarity (Canonical vs Bipolar): {:.4f}\n", mean_sim_bipolar);
    std::cout << std::format("  • Mean Latent Cosine Similarity (Canonical vs Ordinal): {:.4f}\n", mean_sim_ordinal);
    std::cout << std::format("  • Cross-Syntax Choice Agreement Rate: {:.2f}%\n", agreement_bipolar * 100.0);

    ACCEPTANCE_ASSERT(mean_sim_bipolar >= 0.80, "Mean latent cosine similarity (Canonical vs Bipolar) must be >= 0.80");
    ACCEPTANCE_ASSERT(mean_sim_ordinal >= 0.50, "Mean latent cosine similarity (Canonical vs Ordinal) must be >= 0.50");
    ACCEPTANCE_ASSERT(agreement_bipolar >= 0.90, "Cross-syntax choice agreement rate must be >= 90%");

    std::cout << "  ✓ Scientific acceptance criteria passed for invariant latent manifolds!\n";
}

int main() {
    std::cout << "=== TinySystemOne Rigorous Scientific Acceptance Tests ===\n";
    test_disjoint_partitions();
    test_generalization_and_ood_energy();
    test_semantic_invariance_and_latent_manifolds();
    std::cout << "\033[1;32mAll acceptance criteria satisfied with zero data leakage!\033[0m\n";
    return 0;
}
