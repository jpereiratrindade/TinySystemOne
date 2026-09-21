#pragma once

#include "tensor.hpp"
#include "random.hpp"
#include <string>
#include <string_view>
#include <vector>
#include <tuple>
#include <format>

namespace tso {

enum class Choice : std::size_t {
    Nominal = 0,
    Degraded = 1,
    Inconsistent = 2,
    Unknown = 3
};

inline std::string_view to_string(Choice c) {
    switch (c) {
        case Choice::Nominal: return "NOMINAL";
        case Choice::Degraded: return "DEGRADED";
        case Choice::Inconsistent: return "INCONSISTENT";
        case Choice::Unknown: return "UNKNOWN";
    }
    return "INVALID";
}

enum class Noul : std::size_t {
    None = 0,
    Declaration = 1,
    Runtime = 2,
    Witness = 3,
    Freshness = 4,
    Health = 5,
    Multiple = 6
};

inline std::string_view to_string(Noul n) {
    switch (n) {
        case Noul::None: return "NONE";
        case Noul::Declaration: return "DECLARATION";
        case Noul::Runtime: return "RUNTIME";
        case Noul::Witness: return "WITNESS";
        case Noul::Freshness: return "FRESHNESS";
        case Noul::Health: return "HEALTH";
        case Noul::Multiple: return "MULTIPLE";
    }
    return "INVALID";
}

enum class RuntimeState : std::size_t { Running = 0, Absent = 1, Unknown = 2 };
enum class WitnessState : std::size_t { Valid = 0, Invalid = 1, Stale = 2, Unknown = 3 };
enum class FreshnessState : std::size_t { Fresh = 0, Aging = 1, Expired = 2 };
enum class HealthState : std::size_t { Healthy = 0, Degraded = 1, Failing = 2, Unknown = 3 };

struct TripleJudgment {
    Choice choice{Choice::Nominal};
    Noul noul{Noul::None};
    Scalar score{1.0f};
};

struct StructuredState {
    bool declared{true};
    bool registered{true};
    RuntimeState runtime{RuntimeState::Running};
    WitnessState witness{WitnessState::Valid};
    FreshnessState freshness{FreshnessState::Fresh};
    HealthState health{HealthState::Healthy};

    [[nodiscard]] Vector encode() const {
        Vector x(18, 0.0f);
        // declared (2)
        x[declared ? 0 : 1] = 1.0f;
        // registered (2)
        x[2 + (registered ? 0 : 1)] = 1.0f;
        // runtime (3)
        x[4 + static_cast<std::size_t>(runtime)] = 1.0f;
        // witness (4)
        x[7 + static_cast<std::size_t>(witness)] = 1.0f;
        // freshness (3)
        x[11 + static_cast<std::size_t>(freshness)] = 1.0f;
        // health (4)
        x[14 + static_cast<std::size_t>(health)] = 1.0f;
        return x;
    }

    [[nodiscard]] Choice ground_truth() const {
        return evaluate_judgment().choice;
    }

    [[nodiscard]] TripleJudgment evaluate_judgment() const {
        // 1. Inconsistent (Contradictory evidence)
        if (!declared && registered) {
            return {Choice::Inconsistent, Noul::Declaration, 0.05f};
        }
        if (runtime == RuntimeState::Running && witness == WitnessState::Invalid) {
            return {Choice::Inconsistent, Noul::Witness, 0.05f};
        }
        if (runtime == RuntimeState::Absent && witness == WitnessState::Valid) {
            return {Choice::Inconsistent, Noul::Runtime, 0.05f};
        }
        if (health == HealthState::Healthy && freshness == FreshnessState::Expired) {
            return {Choice::Inconsistent, Noul::Freshness, 0.05f};
        }

        // 2. Unknown (Insufficient evidence)
        std::size_t unknown_count = 0;
        if (runtime == RuntimeState::Unknown) ++unknown_count;
        if (witness == WitnessState::Unknown) ++unknown_count;
        if (health == HealthState::Unknown) ++unknown_count;
        if (unknown_count >= 2 || (runtime == RuntimeState::Unknown && witness == WitnessState::Unknown)) {
            return {Choice::Unknown, Noul::Multiple, 0.15f};
        }

        // 3. Degraded
        if (health == HealthState::Failing) {
            return {Choice::Degraded, Noul::Health, 0.15f};
        }
        if (runtime == RuntimeState::Absent) {
            return {Choice::Degraded, Noul::Runtime, 0.25f};
        }
        if (freshness == FreshnessState::Expired) {
            return {Choice::Degraded, Noul::Freshness, 0.35f};
        }
        if (witness == WitnessState::Stale) {
            return {Choice::Degraded, Noul::Witness, 0.45f};
        }
        if (health == HealthState::Degraded) {
            return {Choice::Degraded, Noul::Health, 0.55f};
        }

        // 4. Nominal
        if (declared && registered && runtime == RuntimeState::Running &&
            witness == WitnessState::Valid && health == HealthState::Healthy) {
            const Scalar s = (freshness == FreshnessState::Aging) ? 0.90f : 1.00f;
            return {Choice::Nominal, Noul::None, s};
        }

        return {Choice::Degraded, Noul::Multiple, 0.50f};
    }

    [[nodiscard]] std::string to_json_str() const {
        auto rt_str = [](RuntimeState r) {
            switch (r) {
                case RuntimeState::Running: return "running";
                case RuntimeState::Absent: return "absent";
                case RuntimeState::Unknown: return "unknown";
            }
            return "";
        };
        auto wt_str = [](WitnessState w) {
            switch (w) {
                case WitnessState::Valid: return "valid";
                case WitnessState::Invalid: return "invalid";
                case WitnessState::Stale: return "stale";
                case WitnessState::Unknown: return "unknown";
            }
            return "";
        };
        auto fr_str = [](FreshnessState f) {
            switch (f) {
                case FreshnessState::Fresh: return "fresh";
                case FreshnessState::Aging: return "aging";
                case FreshnessState::Expired: return "expired";
            }
            return "";
        };
        auto hl_str = [](HealthState h) {
            switch (h) {
                case HealthState::Healthy: return "healthy";
                case HealthState::Degraded: return "degraded";
                case HealthState::Failing: return "failing";
                case HealthState::Unknown: return "unknown";
            }
            return "";
        };

        return std::format(
            "{{\"declared\":{},\"registered\":{},\"runtime\":\"{}\",\"witness\":\"{}\",\"freshness\":\"{}\",\"health\":\"{}\"}}",
            declared ? "true" : "false",
            registered ? "true" : "false",
            rt_str(runtime),
            wt_str(witness),
            fr_str(freshness),
            hl_str(health)
        );
    }
};

struct DataSample {
    Vector x;
    std::size_t y;            // Choice target class
    std::size_t noul_target;  // Noul target class
    Scalar score_target;      // Continuous score target [0, 1]
    std::string description;
};

struct DatasetSplit {
    std::vector<DataSample> train;
    std::vector<DataSample> val;
    std::vector<DataSample> test;
    std::vector<DataSample> ood;
};

class DatasetGenerator {
public:
    static StructuredState generate_sample_for_class(Choice target_choice, Random& rng) {
        StructuredState state;
        while (true) {
            switch (target_choice) {
                case Choice::Nominal:
                    state.declared = true;
                    state.registered = true;
                    state.runtime = RuntimeState::Running;
                    state.witness = WitnessState::Valid;
                    state.freshness = (rng.uniform() > 0.4f) ? FreshnessState::Fresh : FreshnessState::Aging;
                    state.health = HealthState::Healthy;
                    break;
                case Choice::Degraded: {
                    state.declared = true;
                    state.registered = true;
                    state.runtime = RuntimeState::Running;
                    state.witness = WitnessState::Valid;
                    state.freshness = FreshnessState::Fresh;
                    state.health = HealthState::Healthy;
                    const int deg_type = rng.uniform_int(0, 4);
                    if (deg_type == 0) state.health = HealthState::Degraded;
                    else if (deg_type == 1) state.witness = WitnessState::Stale;
                    else if (deg_type == 2) state.freshness = FreshnessState::Expired;
                    else if (deg_type == 3) state.runtime = RuntimeState::Absent;
                    else state.health = HealthState::Failing;
                    break;
                }
                case Choice::Inconsistent: {
                    const int scenario = rng.uniform_int(0, 3);
                    if (scenario == 0) {
                        state.declared = false;
                        state.registered = true; // contradictory
                        state.runtime = RuntimeState::Running;
                        state.witness = WitnessState::Valid;
                    } else if (scenario == 1) {
                        state.declared = true;
                        state.registered = true;
                        state.runtime = RuntimeState::Running;
                        state.witness = WitnessState::Invalid; // contradictory
                    } else if (scenario == 2) {
                        state.declared = true;
                        state.registered = true;
                        state.runtime = RuntimeState::Absent;
                        state.witness = WitnessState::Valid; // contradictory
                    } else {
                        state.declared = true;
                        state.registered = true;
                        state.health = HealthState::Healthy;
                        state.freshness = FreshnessState::Expired; // contradictory
                    }
                    break;
                }
                case Choice::Unknown:
                    state.declared = (rng.uniform() > 0.5f);
                    state.registered = (rng.uniform() > 0.5f);
                    state.runtime = RuntimeState::Unknown;
                    state.witness = (RuntimeState::Unknown == state.runtime && rng.uniform() > 0.3f) 
                                    ? WitnessState::Unknown : WitnessState::Stale;
                    state.health = HealthState::Unknown;
                    break;
            }

            if (state.ground_truth() == target_choice) {
                return state;
            }
        }
    }

    static DatasetSplit generate_exp001(std::size_t num_samples_per_class = 250, std::uint64_t seed = 42) {
        Random rng(seed);
        std::vector<DataSample> balanced;

        for (std::size_t c = 0; c < 4; ++c) {
            Choice target_choice = static_cast<Choice>(c);
            for (std::size_t i = 0; i < num_samples_per_class; ++i) {
                StructuredState state = generate_sample_for_class(target_choice, rng);
                TripleJudgment j = state.evaluate_judgment();
                balanced.push_back({
                    .x = state.encode(),
                    .y = static_cast<std::size_t>(j.choice),
                    .noul_target = static_cast<std::size_t>(j.noul),
                    .score_target = j.score,
                    .description = state.to_json_str()
                });
            }
        }

        rng.shuffle(balanced);

        // Partition 80% train, 10% val, 10% test
        const std::size_t total = balanced.size();
        const std::size_t train_size = (total * 80) / 100;
        const std::size_t val_size = (total * 10) / 100;

        DatasetSplit split;
        for (std::size_t i = 0; i < total; ++i) {
            if (i < train_size) {
                split.train.push_back(balanced[i]);
            } else if (i < train_size + val_size) {
                split.val.push_back(balanced[i]);
            } else {
                split.test.push_back(balanced[i]);
            }
        }

        // Generate Out-Of-Distribution (OOD) and extreme ambiguity cases
        // 1. All zero vector (No signal)
        split.ood.push_back({
            .x = Vector(18, 0.0f),
            .y = static_cast<std::size_t>(Choice::Unknown),
            .noul_target = static_cast<std::size_t>(Noul::Multiple),
            .score_target = 0.0f,
            .description = "{\"ood\":\"all_zeros_no_signal\"}"
        });

        // 2. Uniform noise vector
        Vector uniform_v(18, 1.0f / 18.0f);
        split.ood.push_back({
            .x = uniform_v,
            .y = static_cast<std::size_t>(Choice::Unknown),
            .noul_target = static_cast<std::size_t>(Noul::Multiple),
            .score_target = 0.0f,
            .description = "{\"ood\":\"uniform_dispersion\"}"
        });

        // 3. Multi-hot contradictory conflict: multiple states active simultaneously
        Vector multihot_v(18, 0.0f);
        multihot_v[0] = 1.0f; multihot_v[1] = 1.0f; // declared true AND false
        multihot_v[4] = 1.0f; multihot_v[5] = 1.0f; // running AND absent
        multihot_v[7] = 1.0f; multihot_v[8] = 1.0f; // valid AND invalid
        split.ood.push_back({
            .x = multihot_v,
            .y = static_cast<std::size_t>(Choice::Inconsistent),
            .noul_target = static_cast<std::size_t>(Noul::Multiple),
            .score_target = 0.0f,
            .description = "{\"ood\":\"simultaneous_multihot_conflict\"}"
        });

        // 4. Random Gaussian corrupted vectors
        for (int k = 0; k < 10; ++k) {
            Vector corrupt_v(18);
            for (auto& val : corrupt_v) val = std::abs(rng.normal(0.5f, 0.3f));
            split.ood.push_back({
                .x = corrupt_v,
                .y = static_cast<std::size_t>(Choice::Unknown),
                .noul_target = static_cast<std::size_t>(Noul::Multiple),
                .score_target = 0.0f,
                .description = std::format("{{\"ood\":\"corrupted_gaussian_sample_{}\"}}", k)
            });
        }

        return split;
    }
};

} // namespace tso
