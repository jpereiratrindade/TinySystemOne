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

void test_canonical_universe() {
    std::cout << "[TEST] Running test_canonical_universe...\n";
    auto universe = tso::DatasetGenerator::generate_canonical_universe();
    TSO_ASSERT(universe.size() == 576);

    std::unordered_set<std::size_t> unique_ids;
    for (const auto& st : universe) {
        TSO_ASSERT(st.state_id() < 576);
        unique_ids.insert(st.state_id());
    }
    TSO_ASSERT(unique_ids.size() == 576);
    std::cout << "  ✓ Canonical universe generated with exactly 576 unique discrete configurations!\n";
}

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
    TSO_ASSERT(x.size() == 24);
    auto j_nom = nominal_state.evaluate_judgment();
    TSO_ASSERT(j_nom.choice == tso::Choice::Nominal);
    TSO_ASSERT(j_nom.noul == tso::Noul::None);
    TSO_ASSERT(j_nom.score >= 0.99f);
    TSO_ASSERT(j_nom.uncertainty <= 0.05f);

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

    // Missing mask check: 4 fields missing -> Unknown with high uncertainty
    tso::PresenceMask mask;
    mask.declared = false;
    mask.registered = false;
    mask.witness = false;
    mask.health = false;
    auto j_missing = nominal_state.evaluate_judgment_with_mask(mask);
    TSO_ASSERT(j_missing.choice == tso::Choice::Unknown);
    TSO_ASSERT(j_missing.uncertainty >= 0.20f);

    std::cout << "  ✓ Triple judgment and missing mask rules passed!\n";
}

void test_disjoint_dataset_generator() {
    std::cout << "[TEST] Running test_disjoint_dataset_generator...\n";
    auto split = tso::DatasetGenerator::generate_canonical_split(42);

    TSO_ASSERT(!split.train.empty());
    TSO_ASSERT(!split.val.empty());
    TSO_ASSERT(!split.test.empty());
    TSO_ASSERT(!split.ood.empty());

    // Verify zero data leakage
    tso::DatasetGenerator::verify_disjoint_partitions(split);

    for (const auto& sample : split.train) {
        TSO_ASSERT(sample.x.size() == 24);
        TSO_ASSERT(sample.y < 4);
        TSO_ASSERT(sample.noul_target < 7);
        TSO_ASSERT(sample.score_target >= 0.0f && sample.score_target <= 1.0f);
        TSO_ASSERT(sample.uncertainty_target > 0.0f);
    }
    std::cout << "  ✓ Disjoint dataset generator passed (Train: " 
              << split.train.size() << ", Val: " << split.val.size()
              << ", Test: " << split.test.size() << ", OOD: " << split.ood.size() << ") with zero state overlap!\n";
}

int main() {
    std::cout << "=== TinySystemOne Dataset Unit Tests ===\n";
    test_canonical_universe();
    test_state_encoding_and_triple_judgment();
    test_disjoint_dataset_generator();
    std::cout << "All dataset tests passed successfully!\n";
    return 0;
}
