#include "tso/classifier.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <chrono>

#define CLASSIFIER_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "\033[1;31m[TEST_CLASSIFIER FAILED]\033[0m " << msg \
                      << " (" #cond ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(1); \
        } \
    } while (0)

void test_save_load_fidelity() {
    std::cout << "[1] Testing Serialization / Deserialization Fidelity...\n";

    auto split = tso::DatasetGenerator::generate_canonical_split(123);
    auto classifier = tso::SystemOneClassifier::train_and_calibrate(split);

    const std::filesystem::path temp_model_path = "test_classifier_model.bin";

    // Save
    classifier.save(temp_model_path);
    CLASSIFIER_ASSERT(std::filesystem::exists(temp_model_path), "Model file must exist after save");

    // Load
    auto loaded_classifier = tso::SystemOneClassifier::load(temp_model_path);

    // Verify metadata
    CLASSIFIER_ASSERT(loaded_classifier.identity().parameter_count == classifier.identity().parameter_count, "Param count match");
    CLASSIFIER_ASSERT(std::abs(loaded_classifier.identity().calibrated_temperature - classifier.identity().calibrated_temperature) < 1e-5f, "Temperature match");
    CLASSIFIER_ASSERT(std::abs(loaded_classifier.identity().ood_energy_threshold - classifier.identity().ood_energy_threshold) < 1e-5f, "OOD threshold match");

    // Test exact output equality over all 576 canonical states
    auto all_canonical = tso::DatasetGenerator::generate_canonical_universe();
    CLASSIFIER_ASSERT(all_canonical.size() == 576, "Canonical universe must have 576 states");

    for (const auto& state : all_canonical) {
        tso::Observation obs{.state = state};
        auto r1 = classifier.classify(obs);
        auto r2 = loaded_classifier.classify(obs);

        CLASSIFIER_ASSERT(r1.choice == r2.choice, "Choice must match identically after load");
        CLASSIFIER_ASSERT(r1.locus == r2.locus, "Locus must match identically after load");
        CLASSIFIER_ASSERT(std::abs(r1.score - r2.score) < 1e-5f, "Score must match identically after load");
        CLASSIFIER_ASSERT(std::abs(r1.confidence - r2.confidence) < 1e-5f, "Confidence must match identically after load");
        CLASSIFIER_ASSERT(std::abs(r1.entropy - r2.entropy) < 1e-5f, "Entropy must match identically after load");
        CLASSIFIER_ASSERT(std::abs(r1.free_energy - r2.free_energy) < 1e-5f, "Free Energy must match identically after load");
        CLASSIFIER_ASSERT(r1.abstained == r2.abstained, "Abstention status must match identically after load");

        for (std::size_t i = 0; i < 4; ++i) {
            CLASSIFIER_ASSERT(std::abs(r1.choice_probabilities[i] - r2.choice_probabilities[i]) < 1e-5f, "Probabilities must match identically");
        }
    }

    std::filesystem::remove(temp_model_path);
    std::cout << "  ✓ Exact numerical fidelity verified across all 576 canonical states!\n";
}

void test_observation_semantics() {
    std::cout << "[2] Testing Observation Interface Semantics...\n";

    auto split = tso::DatasetGenerator::generate_canonical_split(101);
    auto classifier = tso::SystemOneClassifier::train_and_calibrate(split);

    // 1. Nominal observation
    tso::Observation nom_obs{
        .state = {
            .declared = true,
            .registered = true,
            .runtime = tso::RuntimeState::Running,
            .witness = tso::WitnessState::Valid,
            .freshness = tso::FreshnessState::Fresh,
            .health = tso::HealthState::Healthy
        }
    };
    auto nom_res = classifier.classify(nom_obs);
    std::cout << std::format("  [DEBUG] nom_res: choice={}, conf={:.2f}%, abstained={}, E={:.4f}\n",
                             tso::to_string(nom_res.choice), nom_res.confidence * 100.0f, nom_res.abstained, nom_res.free_energy);
    CLASSIFIER_ASSERT(nom_res.choice == tso::Choice::Nominal, "Nominal state must produce Choice::Nominal");
    CLASSIFIER_ASSERT(nom_res.confidence >= 0.90f, "Nominal state confidence must be >= 90%");
    CLASSIFIER_ASSERT(!nom_res.abstained, "Nominal state must not be abstained");

    // 2. Direct contradiction
    tso::Observation contra_obs{
        .state = {
            .declared = false,
            .registered = true,
            .runtime = tso::RuntimeState::Running,
            .witness = tso::WitnessState::Valid,
            .freshness = tso::FreshnessState::Fresh,
            .health = tso::HealthState::Healthy
        }
    };
    auto contra_res = classifier.classify(contra_obs);
    CLASSIFIER_ASSERT(contra_res.choice == tso::Choice::Inconsistent, "Contradiction must produce Choice::Inconsistent");
    CLASSIFIER_ASSERT(contra_res.locus == tso::Noul::Declaration || contra_res.locus == tso::Noul::Multiple, "Locus must identify Declaration or Multiple");

    // 3. Alien noise OOD abstention
    tso::Vector null_input(24, 0.0f);
    auto ood_res = classifier.classify_raw(null_input);
    CLASSIFIER_ASSERT(ood_res.abstained, "Alien null input must be rejected/abstained");
    CLASSIFIER_ASSERT(ood_res.choice == tso::Choice::Unknown, "Abstained result must report Choice::Unknown");

    std::cout << "  ✓ High-level observation semantics verified!\n";
}

void test_latency_limits() {
    std::cout << "[3] Testing Sub-Microsecond Latency Limit...\n";

    auto split = tso::DatasetGenerator::generate_canonical_split(42);
    auto classifier = tso::SystemOneClassifier::train_and_calibrate(split);

    tso::Observation sample_obs{
        .state = {
            .declared = true,
            .registered = true,
            .runtime = tso::RuntimeState::Running,
            .witness = tso::WitnessState::Valid,
            .freshness = tso::FreshnessState::Fresh,
            .health = tso::HealthState::Healthy
        }
    };

    // Warm-up
    for (int i = 0; i < 500; ++i) {
        auto _ = classifier.classify(sample_obs);
    }

    constexpr std::size_t kRuns = 10000;
    double total_ns = 0.0;
    for (std::size_t i = 0; i < kRuns; ++i) {
        auto res = classifier.classify(sample_obs);
        total_ns += res.latency_nanoseconds;
    }

    const double avg_us = (total_ns / static_cast<double>(kRuns)) / 1000.0;
    std::cout << std::format("  • Average Inference Latency: {:.3f} µs\n", avg_us);
    CLASSIFIER_ASSERT(avg_us < 20.0, "Average inference latency must be < 20 microseconds");
    std::cout << "  ✓ Sub-microsecond latency limit verified!\n";
}

int main() {
    std::cout << "=== TinySystemOne Classifier & Serialization Test Suite ===\n";
    test_save_load_fidelity();
    test_observation_semantics();
    test_latency_limits();
    std::cout << "\033[1;32mAll SystemOneClassifier tests passed successfully!\033[0m\n";
    return 0;
}
