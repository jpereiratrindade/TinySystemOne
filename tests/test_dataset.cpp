#include <tso/dataset.hpp>
#include <iostream>
#include <cstdlib>

#define TSO_ASSERT(expr) \
    do { \
        if (!(expr)) { \
            std::cerr << "Assertion failed: " #expr " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::abort(); \
        } \
    } while (0)

void test_state_encoding_and_triple_judgment() {
    std::cout << "[TEST] Running test_state_encoding_and_triple_judgment...\n";
    tso::StructuredState nominal_state{
        .declared = true,
        .registered = true,
        .runtime = tso::RuntimeState::Running,
        .witness = tso::WitnessState::Valid,
        .freshness = tso::FreshnessState::Fresh,
        .health = tso::HealthState::Healthy
    };

    tso::Vector x = nominal_state.encode();
    TSO_ASSERT(x.size() == 18);
    auto j_nom = nominal_state.evaluate_judgment();
    TSO_ASSERT(j_nom.choice == tso::Choice::Nominal);
    TSO_ASSERT(j_nom.noul == tso::Noul::None);
    TSO_ASSERT(j_nom.score >= 0.99f);

    // Inconsistent state check (declaration mismatch)
    tso::StructuredState inconsistent_decl{
        .declared = false,
        .registered = true,
        .runtime = tso::RuntimeState::Running,
        .witness = tso::WitnessState::Valid,
        .freshness = tso::FreshnessState::Fresh,
        .health = tso::HealthState::Healthy
    };
    auto j_inc = inconsistent_decl.evaluate_judgment();
    TSO_ASSERT(j_inc.choice == tso::Choice::Inconsistent);
    TSO_ASSERT(j_inc.noul == tso::Noul::Declaration);
    TSO_ASSERT(j_inc.score <= 0.1f);

    // Degraded state check (witness stale)
    tso::StructuredState degraded_witness{
        .declared = true,
        .registered = true,
        .runtime = tso::RuntimeState::Running,
        .witness = tso::WitnessState::Stale,
        .freshness = tso::FreshnessState::Fresh,
        .health = tso::HealthState::Healthy
    };
    auto j_deg = degraded_witness.evaluate_judgment();
    TSO_ASSERT(j_deg.choice == tso::Choice::Degraded);
    TSO_ASSERT(j_deg.noul == tso::Noul::Witness);
    TSO_ASSERT(j_deg.score >= 0.4f && j_deg.score <= 0.6f);

    std::cout << "  ✓ Triple judgment rules passed!\n";
}

void test_dataset_generator() {
    std::cout << "[TEST] Running test_dataset_generator...\n";
    auto split = tso::DatasetGenerator::generate_exp001(100, 42);

    TSO_ASSERT(!split.train.empty());
    TSO_ASSERT(!split.val.empty());
    TSO_ASSERT(!split.test.empty());
    TSO_ASSERT(!split.ood.empty());

    for (const auto& sample : split.train) {
        TSO_ASSERT(sample.x.size() == 18);
        TSO_ASSERT(sample.y < 4);
        TSO_ASSERT(sample.noul_target < 7);
        TSO_ASSERT(sample.score_target >= 0.0f && sample.score_target <= 1.0f);
    }
    std::cout << "  ✓ Dataset generator split passed (Train: " 
              << split.train.size() << ", Val: " << split.val.size()
              << ", Test: " << split.test.size() << ", OOD: " << split.ood.size() << ")!\n";
}

int main() {
    std::cout << "=== TinySystemOne Dataset Unit Tests ===\n";
    test_state_encoding_and_triple_judgment();
    test_dataset_generator();
    std::cout << "All dataset tests passed successfully!\n";
    return 0;
}
