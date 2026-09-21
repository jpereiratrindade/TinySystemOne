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

struct PresenceMask {
    bool declared{true};
    bool registered{true};
    bool runtime{true};
    bool witness{true};
    bool freshness{true};
    bool health{true};

    [[nodiscard]] std::size_t count_missing() const {
        std::size_t m = 0;
        if (!declared) ++m;
        if (!registered) ++m;
        if (!runtime) ++m;
        if (!witness) ++m;
        if (!freshness) ++m;
        if (!health) ++m;
        return m;
    }
};

struct TripleJudgment {
    Choice choice{Choice::Nominal};
    Noul noul{Noul::None};
    Scalar score{1.0f};
    Scalar uncertainty{0.01f}; // Epistemic variance target
};

struct StructuredState {
    bool declared{true};
    bool registered{true};
    RuntimeState runtime{RuntimeState::Running};
    WitnessState witness{WitnessState::Valid};
    FreshnessState freshness{FreshnessState::Fresh};
    HealthState health{HealthState::Healthy};

    [[nodiscard]] Vector encode() const {
        return encode_with_mask(PresenceMask{});
    }

    [[nodiscard]] Vector encode_with_mask(const PresenceMask& mask) const {
        // 18 feature dims + 6 presence mask dims = 24 dimensions
        Vector x(24, 0.0f);
        
        // Features (only set if presence flag is true)
        if (mask.declared) x[declared ? 0 : 1] = 1.0f;
        if (mask.registered) x[2 + (registered ? 0 : 1)] = 1.0f;
        if (mask.runtime) x[4 + static_cast<std::size_t>(runtime)] = 1.0f;
        if (mask.witness) x[7 + static_cast<std::size_t>(witness)] = 1.0f;
        if (mask.freshness) x[11 + static_cast<std::size_t>(freshness)] = 1.0f;
        if (mask.health) x[14 + static_cast<std::size_t>(health)] = 1.0f;

        // Presence Mask indicators
        x[18] = mask.declared ? 1.0f : 0.0f;
        x[19] = mask.registered ? 1.0f : 0.0f;
        x[20] = mask.runtime ? 1.0f : 0.0f;
        x[21] = mask.witness ? 1.0f : 0.0f;
        x[22] = mask.freshness ? 1.0f : 0.0f;
        x[23] = mask.health ? 1.0f : 0.0f;

        return x;
    }

    [[nodiscard]] Choice ground_truth() const {
        return evaluate_judgment().choice;
    }

    [[nodiscard]] TripleJudgment evaluate_judgment() const {
        return evaluate_judgment_with_mask(PresenceMask{});
    }

    [[nodiscard]] TripleJudgment evaluate_judgment_with_mask(const PresenceMask& mask) const {
        const std::size_t missing_cnt = mask.count_missing();

        // If information is severely missing (>= 3 fields missing)
        if (missing_cnt >= 3) {
            const Scalar unc = 0.05f + static_cast<Scalar>(missing_cnt) * 0.05f;
            return {Choice::Unknown, Noul::Multiple, 0.20f, unc};
        }

        // 1. Inconsistent (Contradictory evidence)
        if (mask.declared && mask.registered && !declared && registered) {
            return {Choice::Inconsistent, Noul::Declaration, 0.05f, 0.01f + 0.04f * static_cast<Scalar>(missing_cnt)};
        }
        if (mask.runtime && mask.witness && runtime == RuntimeState::Running && witness == WitnessState::Invalid) {
            return {Choice::Inconsistent, Noul::Witness, 0.05f, 0.01f + 0.04f * static_cast<Scalar>(missing_cnt)};
        }
        if (mask.runtime && mask.witness && runtime == RuntimeState::Absent && witness == WitnessState::Valid) {
            return {Choice::Inconsistent, Noul::Runtime, 0.05f, 0.01f + 0.04f * static_cast<Scalar>(missing_cnt)};
        }
        if (mask.health && mask.freshness && health == HealthState::Healthy && freshness == FreshnessState::Expired) {
            return {Choice::Inconsistent, Noul::Freshness, 0.05f, 0.01f + 0.04f * static_cast<Scalar>(missing_cnt)};
        }

        // 2. Unknown (Insufficient evidence via states or missingness)
        std::size_t unknown_count = missing_cnt;
        if (mask.runtime && runtime == RuntimeState::Unknown) ++unknown_count;
        if (mask.witness && witness == WitnessState::Unknown) ++unknown_count;
        if (mask.health && health == HealthState::Unknown) ++unknown_count;
        if (unknown_count >= 2) {
            const Scalar unc = 0.02f + static_cast<Scalar>(unknown_count) * 0.04f;
            return {Choice::Unknown, Noul::Multiple, 0.15f, unc};
        }

        // 3. Degraded
        if (mask.health && health == HealthState::Failing) {
            return {Choice::Degraded, Noul::Health, 0.15f, 0.02f};
        }
        if (mask.runtime && runtime == RuntimeState::Absent) {
            return {Choice::Degraded, Noul::Runtime, 0.25f, 0.02f};
        }
        if (mask.freshness && freshness == FreshnessState::Expired) {
            return {Choice::Degraded, Noul::Freshness, 0.35f, 0.02f};
        }
        if (mask.witness && witness == WitnessState::Stale) {
            return {Choice::Degraded, Noul::Witness, 0.45f, 0.02f};
        }
        if (mask.health && health == HealthState::Degraded) {
            return {Choice::Degraded, Noul::Health, 0.55f, 0.02f};
        }

        // 4. Nominal
        if (declared && registered && runtime == RuntimeState::Running &&
            witness == WitnessState::Valid && health == HealthState::Healthy) {
            const Scalar s = (freshness == FreshnessState::Aging) ? 0.90f : 1.00f;
            const Scalar unc = 0.01f + 0.03f * static_cast<Scalar>(missing_cnt);
            return {Choice::Nominal, Noul::None, s, unc};
        }

        return {Choice::Degraded, Noul::Multiple, 0.50f, 0.05f};
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
    Scalar uncertainty_target;// Uncertainty variance target
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
                    .uncertainty_target = j.uncertainty,
                    .description = state.to_json_str()
                });
            }
        }

        rng.shuffle(balanced);

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

        // OOD Samples (24-dimensional vectors)
        split.ood.push_back({
            .x = Vector(24, 0.0f),
            .y = static_cast<std::size_t>(Choice::Unknown),
            .noul_target = static_cast<std::size_t>(Noul::Multiple),
            .score_target = 0.0f,
            .uncertainty_target = 0.5f,
            .description = "{\"ood\":\"all_zeros_no_signal\"}"
        });

        Vector uniform_v(24, 1.0f / 24.0f);
        split.ood.push_back({
            .x = uniform_v,
            .y = static_cast<std::size_t>(Choice::Unknown),
            .noul_target = static_cast<std::size_t>(Noul::Multiple),
            .score_target = 0.0f,
            .uncertainty_target = 0.5f,
            .description = "{\"ood\":\"uniform_dispersion\"}"
        });

        Vector multihot_v(24, 0.0f);
        multihot_v[0] = 1.0f; multihot_v[1] = 1.0f;
        multihot_v[4] = 1.0f; multihot_v[5] = 1.0f;
        multihot_v[7] = 1.0f; multihot_v[8] = 1.0f;
        for (std::size_t k = 18; k < 24; ++k) multihot_v[k] = 1.0f;
        split.ood.push_back({
            .x = multihot_v,
            .y = static_cast<std::size_t>(Choice::Inconsistent),
            .noul_target = static_cast<std::size_t>(Noul::Multiple),
            .score_target = 0.0f,
            .uncertainty_target = 0.3f,
            .description = "{\"ood\":\"simultaneous_multihot_conflict\"}"
        });

        return split;
    }

    // EXP-003: Training with missingness/dropout on presence mask
    static DatasetSplit generate_exp003(std::size_t num_samples_per_class = 300, std::uint64_t seed = 42) {
        Random rng(seed);
        std::vector<DataSample> samples;

        for (std::size_t c = 0; c < 4; ++c) {
            Choice target_choice = static_cast<Choice>(c);
            for (std::size_t i = 0; i < num_samples_per_class; ++i) {
                StructuredState state = generate_sample_for_class(target_choice, rng);

                // Apply probabilistic missingness during training (0% to 50% chance per field)
                PresenceMask mask;
                const float missing_rate = rng.uniform(0.0f, 0.4f);
                mask.declared = (rng.uniform() > missing_rate);
                mask.registered = (rng.uniform() > missing_rate);
                mask.runtime = (rng.uniform() > missing_rate);
                mask.witness = (rng.uniform() > missing_rate);
                mask.freshness = (rng.uniform() > missing_rate);
                mask.health = (rng.uniform() > missing_rate);

                TripleJudgment j = state.evaluate_judgment_with_mask(mask);
                samples.push_back({
                    .x = state.encode_with_mask(mask),
                    .y = static_cast<std::size_t>(j.choice),
                    .noul_target = static_cast<std::size_t>(j.noul),
                    .score_target = j.score,
                    .uncertainty_target = j.uncertainty,
                    .description = state.to_json_str()
                });
            }
        }

        rng.shuffle(samples);

        const std::size_t total = samples.size();
        const std::size_t train_size = (total * 80) / 100;
        const std::size_t val_size = (total * 10) / 100;

        DatasetSplit split;
        for (std::size_t i = 0; i < total; ++i) {
            if (i < train_size) {
                split.train.push_back(samples[i]);
            } else if (i < train_size + val_size) {
                split.val.push_back(samples[i]);
            } else {
                split.test.push_back(samples[i]);
            }
        }

        return split;
    }
};

} // namespace tso
