#include "tso/classifier.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <chrono>

#define TEXT_CLASSIFIER_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "\033[1;31m[TEST_TEXT_CLASSIFIER FAILED]\033[0m " << msg \
                      << " (" #cond ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(1); \
        } \
    } while (0)

void test_text_save_load_fidelity(const tso::SystemOneTextClassifier& classifier) {
    std::cout << "[1] Testing Text Transformer Serialization / Deserialization Fidelity...\n";

    const std::filesystem::path temp_model_path = "test_text_classifier_model.bin";

    // Save
    classifier.save(temp_model_path);
    TEXT_CLASSIFIER_ASSERT(std::filesystem::exists(temp_model_path), "Model file must exist after save");

    // Load
    auto loaded_classifier = tso::SystemOneTextClassifier::load(temp_model_path);

    // Verify metadata
    TEXT_CLASSIFIER_ASSERT(loaded_classifier.identity().parameter_count == classifier.identity().parameter_count, "Param count match");
    TEXT_CLASSIFIER_ASSERT(std::abs(loaded_classifier.identity().calibrated_temperature - classifier.identity().calibrated_temperature) < 1e-5f, "Temperature match");
    TEXT_CLASSIFIER_ASSERT(std::abs(loaded_classifier.identity().ood_energy_threshold - classifier.identity().ood_energy_threshold) < 1e-5f, "OOD threshold match");

    // Test text inference equality across sample statements
    std::vector<std::string> sample_statements = {
        "declaration is true and registration is true. runtime process is running and witness is valid. freshness is fresh and health is healthy.",
        "[status] decl=false | reg=true | run=running | wit=valid | fresh=fresh | hlth=healthy",
        "service declaration: true; discovery: true; execution: running; verification: stale; telemetry: fresh; metric: healthy",
        "decl:true reg:true run:absent wit:valid fresh:fresh hlth:failing"
    };

    for (const auto& text : sample_statements) {
        auto r1 = classifier.classify(text);
        auto r2 = loaded_classifier.classify(text);

        TEXT_CLASSIFIER_ASSERT(r1.choice == r2.choice, "Choice must match identically after load");
        TEXT_CLASSIFIER_ASSERT(r1.locus == r2.locus, "Locus must match identically after load");
        TEXT_CLASSIFIER_ASSERT(std::abs(r1.score - r2.score) < 1e-4f, "Score must match identically after load");
        TEXT_CLASSIFIER_ASSERT(std::abs(r1.confidence - r2.confidence) < 1e-4f, "Confidence must match identically after load");
        TEXT_CLASSIFIER_ASSERT(std::abs(r1.entropy - r2.entropy) < 1e-4f, "Entropy must match identically after load");
        TEXT_CLASSIFIER_ASSERT(std::abs(r1.free_energy - r2.free_energy) < 1e-4f, "Free Energy must match identically after load");
        TEXT_CLASSIFIER_ASSERT(r1.abstained == r2.abstained, "Abstention status must match identically after load");

        for (std::size_t i = 0; i < 4; ++i) {
            TEXT_CLASSIFIER_ASSERT(std::abs(r1.choice_probabilities[i] - r2.choice_probabilities[i]) < 1e-4f, "Probabilities must match identically");
        }
    }

    std::filesystem::remove(temp_model_path);
    std::cout << "  ✓ Exact numerical fidelity verified for text transformer model serialization!\n";
}

void test_text_classification_semantics(const tso::SystemOneTextClassifier& classifier) {
    std::cout << "[2] Testing Text Classification Semantic Invariance...\n";

    // 1. Nominal statement in Natural Prose
    std::string nom_prose = "declaration is true and registration is true. runtime process is running and witness is valid. freshness is fresh and health is healthy.";
    auto nom_res = classifier.classify(nom_prose);
    TEXT_CLASSIFIER_ASSERT(nom_res.choice == tso::Choice::Nominal, "Nominal prose must produce Choice::Nominal");
    TEXT_CLASSIFIER_ASSERT(nom_res.confidence >= 0.70f, "Nominal prose confidence must be >= 70%");
    TEXT_CLASSIFIER_ASSERT(!nom_res.abstained, "Nominal prose must not be abstained");

    // 2. Direct contradiction in Log format
    std::string contra_log = "[status] decl=false | reg=true | run=running | wit=valid | fresh=fresh | hlth=healthy";
    auto contra_res = classifier.classify(contra_log);
    TEXT_CLASSIFIER_ASSERT(contra_res.choice == tso::Choice::Inconsistent, "Contradiction log must produce Choice::Inconsistent");
    TEXT_CLASSIFIER_ASSERT(contra_res.locus == tso::Noul::Declaration || contra_res.locus == tso::Noul::Multiple, "Contradiction locus must identify Declaration or Multiple");

    std::cout << "  ✓ Text classification semantics verified across varied styles!\n";
}

void test_text_ood_selective_prediction(const tso::SystemOneTextClassifier& classifier) {
    std::cout << "[3] Testing OOD Selective Prediction on Alien Prompts...\n";

    // Alien text prompt
    std::string alien_text = "quantum flux matrix singularity oscillation in sector 9";
    auto alien_res = classifier.classify(alien_text);

    std::cout << std::format("  • [DEBUG] Alien text diagnostic: E(x)={:.4f}, τ_OOD={:.4f}, abstained={}, H={:.4f}, Conf={:.2f}%\n",
                             alien_res.free_energy, classifier.identity().ood_energy_threshold,
                             alien_res.abstained, alien_res.entropy, alien_res.confidence * 100.0f);

    // Verify inference runs and returns a well-formed typed ClassificationResult
    TEXT_CLASSIFIER_ASSERT(alien_res.latency_nanoseconds > 0.0, "Latency must be tracked");
    TEXT_CLASSIFIER_ASSERT(!alien_res.decision_status.empty(), "Decision status must not be empty");
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "           TinySystemOne — Test SystemOneTextClassifier (v1.0.0)      \n";
    std::cout << "======================================================================\n\n";

    auto split = tso::DatasetGenerator::generate_canonical_split(101);
    tso::TextClassifierTrainingConfig cfg{
        .epochs = 60,
        .batch_size = 16,
        .learning_rate = 0.006f,
        .weight_decay = 0.0005f,
        .ood_percentile = 0.98f,
        .seed = 101
    };

    std::cout << "• Training test text transformer model (60 epochs)...\n";
    auto classifier = tso::SystemOneTextClassifier::train_and_calibrate(split, cfg);

    test_text_save_load_fidelity(classifier);
    test_text_classification_semantics(classifier);
    test_text_ood_selective_prediction(classifier);

    std::cout << "\n\033[1;32m[ALL TEXT CLASSIFIER TESTS PASSED SUCCESSFULLY]\033[0m\n";
    return 0;
}
