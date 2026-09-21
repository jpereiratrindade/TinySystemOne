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

void test_state_encoding() {
    std::cout << "[TEST] Running test_state_encoding...\n";
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
    // nominal ground truth check
    TSO_ASSERT(nominal_state.ground_truth() == tso::Choice::Nominal);

    // Inconsistent state check
    tso::StructuredState inconsistent_state{
        .declared = false,
        .registered = true, // contradictory!
        .runtime = tso::RuntimeState::Running,
        .witness = tso::WitnessState::Valid,
        .freshness = tso::FreshnessState::Fresh,
        .health = tso::HealthState::Healthy
    };
    TSO_ASSERT(inconsistent_state.ground_truth() == tso::Choice::Inconsistent);

    // Degraded state check
    tso::StructuredState degraded_state{
        .declared = true,
        .registered = true,
        .runtime = tso::RuntimeState::Running,
        .witness = tso::WitnessState::Valid,
        .freshness = tso::FreshnessState::Expired, // degraded freshness
        .health = tso::HealthState::Degraded
    };
    TSO_ASSERT(degraded_state.ground_truth() == tso::Choice::Degraded);

    // Unknown state check
    tso::StructuredState unknown_state{
        .declared = true,
        .registered = true,
        .runtime = tso::RuntimeState::Unknown,
        .witness = tso::WitnessState::Unknown,
        .freshness = tso::FreshnessState::Fresh,
        .health = tso::HealthState::Unknown
    };
    TSO_ASSERT(unknown_state.ground_truth() == tso::Choice::Unknown);

    std::cout << "  ✓ State encoding & ground truth rules passed!\n";
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
    }
    std::cout << "  ✓ Dataset generator split passed (Train: " 
              << split.train.size() << ", Val: " << split.val.size()
              << ", Test: " << split.test.size() << ", OOD: " << split.ood.size() << ")!\n";
}

int main() {
    std::cout << "=== TinySystemOne Dataset Unit Tests ===\n";
    test_state_encoding();
    test_dataset_generator();
    std::cout << "All dataset tests passed successfully!\n";
    return 0;
}
