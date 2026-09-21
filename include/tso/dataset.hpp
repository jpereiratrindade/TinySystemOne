#pragma once

#include "tensor.hpp"
#include "random.hpp"
#include <string>
#include <string_view>
#include <vector>
#include <tuple>
#include <format>
#include <unordered_set>
#include <stdexcept>

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

enum class OodCategory {
    NoSignal,
    UniformDispersion,
    ContradictoryEvidence,
    NovelCombination,
    CorruptedInput
};

inline std::string_view to_string(OodCategory cat) {
    switch (cat) {
        case OodCategory::NoSignal: return "NO_SIGNAL";
        case OodCategory::UniformDispersion: return "UNIFORM_DISPERSION";
        case OodCategory::ContradictoryEvidence: return "CONTRADICTORY_EVIDENCE";
        case OodCategory::NovelCombination: return "NOVEL_COMBINATION";
        case OodCategory::CorruptedInput: return "CORRUPTED_INPUT";
    }
    return "UNKNOWN";
}

enum class EncodingForm : std::size_t {
    CanonicalOneHot = 0,
    NormalizedOrdinal = 1,
    PermutedChannels = 2,
    BipolarDifferential = 3
};

inline std::string_view to_string(EncodingForm form) {
    switch (form) {
        case EncodingForm::CanonicalOneHot: return "CANONICAL_ONE_HOT";
        case EncodingForm::NormalizedOrdinal: return "NORMALIZED_ORDINAL";
        case EncodingForm::PermutedChannels: return "PERMUTED_CHANNELS";
        case EncodingForm::BipolarDifferential: return "BIPOLAR_DIFFERENTIAL";
    }
    return "UNKNOWN";
}

enum class RuntimeState : std::size_t { Running = 0, Absent = 1, Unknown = 2 };
enum class WitnessState : std::size_t { Valid = 0, Invalid = 1, Stale = 2, Unknown = 3 };
enum class FreshnessState : std::size_t { Fresh = 0, Aging = 1, Expired = 2 };
enum class HealthState : std::size_t { Healthy = 0, Degraded = 1, Failing = 2, Unknown = 3 };

inline std::string_view to_string(RuntimeState s) {
    switch (s) {
        case RuntimeState::Running: return "RUNNING";
        case RuntimeState::Absent: return "ABSENT";
        case RuntimeState::Unknown: return "UNKNOWN";
    }
    return "UNKNOWN";
}

inline std::string_view to_string(WitnessState s) {
    switch (s) {
        case WitnessState::Valid: return "VALID";
        case WitnessState::Invalid: return "INVALID";
        case WitnessState::Stale: return "STALE";
        case WitnessState::Unknown: return "UNKNOWN";
    }
    return "UNKNOWN";
}

inline std::string_view to_string(FreshnessState s) {
    switch (s) {
        case FreshnessState::Fresh: return "FRESH";
        case FreshnessState::Aging: return "AGING";
        case FreshnessState::Expired: return "EXPIRED";
    }
    return "UNKNOWN";
}

inline std::string_view to_string(HealthState s) {
    switch (s) {
        case HealthState::Healthy: return "HEALTHY";
        case HealthState::Degraded: return "DEGRADED";
        case HealthState::Failing: return "FAILING";
        case HealthState::Unknown: return "UNKNOWN";
    }
    return "UNKNOWN";
}

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

    bool operator==(const PresenceMask& other) const = default;
};

struct TripleJudgment {
    Choice choice{Choice::Nominal};
    Noul noul{Noul::None};
    Scalar score{1.0f};         // Learned continuous viability
    Scalar uncertainty{0.01f};   // Epistemic uncertainty target
};

struct StructuredState {
    bool declared{true};
    bool registered{true};
    RuntimeState runtime{RuntimeState::Running};
    WitnessState witness{WitnessState::Valid};
    FreshnessState freshness{FreshnessState::Fresh};
    HealthState health{HealthState::Healthy};

    bool operator==(const StructuredState& other) const = default;

    // Unique integer key identifying this exact state in [0, 575]
    [[nodiscard]] std::size_t state_id() const {
        std::size_t id = 0;
        id = id * 2 + (declared ? 1 : 0);
        id = id * 2 + (registered ? 1 : 0);
        id = id * 3 + static_cast<std::size_t>(runtime);
        id = id * 4 + static_cast<std::size_t>(witness);
        id = id * 3 + static_cast<std::size_t>(freshness);
        id = id * 4 + static_cast<std::size_t>(health);
        return id;
    }

    [[nodiscard]] Vector encode() const {
        return encode_with_mask(PresenceMask{});
    }

    [[nodiscard]] Vector encode_with_mask(const PresenceMask& mask) const {
        Vector x(24, 0.0f);
        
        // 18 feature slots
        if (mask.declared) x[declared ? 0 : 1] = 1.0f;
        if (mask.registered) x[2 + (registered ? 0 : 1)] = 1.0f;
        if (mask.runtime) x[4 + static_cast<std::size_t>(runtime)] = 1.0f;
        if (mask.witness) x[7 + static_cast<std::size_t>(witness)] = 1.0f;
        if (mask.freshness) x[11 + static_cast<std::size_t>(freshness)] = 1.0f;
        if (mask.health) x[14 + static_cast<std::size_t>(health)] = 1.0f;

        // 6 presence mask bits
        x[18] = mask.declared ? 1.0f : 0.0f;
        x[19] = mask.registered ? 1.0f : 0.0f;
        x[20] = mask.runtime ? 1.0f : 0.0f;
        x[21] = mask.witness ? 1.0f : 0.0f;
        x[22] = mask.freshness ? 1.0f : 0.0f;
        x[23] = mask.health ? 1.0f : 0.0f;

        return x;
    }

    [[nodiscard]] Vector encode_as(EncodingForm form, const PresenceMask& mask = {}) const {
        switch (form) {
            case EncodingForm::CanonicalOneHot:
                return encode_with_mask(mask);

            case EncodingForm::NormalizedOrdinal: {
                Vector x(24, 0.0f);
                if (mask.declared) x[0] = declared ? 1.0f : 0.0f;
                if (mask.registered) x[2] = registered ? 1.0f : 0.0f;
                if (mask.runtime) x[4] = static_cast<Scalar>(runtime) / 2.0f;
                if (mask.witness) x[7] = static_cast<Scalar>(witness) / 3.0f;
                if (mask.freshness) x[11] = static_cast<Scalar>(freshness) / 2.0f;
                if (mask.health) x[14] = static_cast<Scalar>(health) / 3.0f;

                x[18] = mask.declared ? 1.0f : 0.0f;
                x[19] = mask.registered ? 1.0f : 0.0f;
                x[20] = mask.runtime ? 1.0f : 0.0f;
                x[21] = mask.witness ? 1.0f : 0.0f;
                x[22] = mask.freshness ? 1.0f : 0.0f;
                x[23] = mask.health ? 1.0f : 0.0f;
                return x;
            }

            case EncodingForm::PermutedChannels: {
                Vector x(24, 0.0f);
                if (mask.health) x[0 + static_cast<std::size_t>(health)] = 1.0f;
                if (mask.freshness) x[4 + static_cast<std::size_t>(freshness)] = 1.0f;
                if (mask.witness) x[7 + static_cast<std::size_t>(witness)] = 1.0f;
                if (mask.runtime) x[11 + static_cast<std::size_t>(runtime)] = 1.0f;
                if (mask.registered) x[14 + (registered ? 0 : 1)] = 1.0f;
                if (mask.declared) x[16 + (declared ? 0 : 1)] = 1.0f;

                x[18] = mask.health ? 1.0f : 0.0f;
                x[19] = mask.freshness ? 1.0f : 0.0f;
                x[20] = mask.witness ? 1.0f : 0.0f;
                x[21] = mask.runtime ? 1.0f : 0.0f;
                x[22] = mask.registered ? 1.0f : 0.0f;
                x[23] = mask.declared ? 1.0f : 0.0f;
                return x;
            }

            case EncodingForm::BipolarDifferential: {
                Vector x = encode_with_mask(mask);
                for (std::size_t i = 0; i < 18; ++i) {
                    x[i] = (x[i] > 0.5f) ? 1.0f : -1.0f;
                }
                return x;
            }
        }
        return encode_with_mask(mask);
    }

    [[nodiscard]] Choice ground_truth() const {
        return evaluate_judgment().choice;
    }

    [[nodiscard]] TripleJudgment evaluate_judgment() const {
        return evaluate_judgment_with_mask(PresenceMask{});
    }

    [[nodiscard]] TripleJudgment evaluate_judgment_with_mask(const PresenceMask& mask) const {
        const std::size_t missing_cnt = mask.count_missing();

        if (missing_cnt >= 3) {
            const Scalar unc = 0.05f + static_cast<Scalar>(missing_cnt) * 0.05f;
            return {Choice::Unknown, Noul::Multiple, 0.20f, unc};
        }

        // 1. Inconsistent (Direct contradiction)
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

        // 2. Unknown (Semantic unknown state or insufficient evidence)
        std::size_t unknown_count = missing_cnt;
        if (mask.runtime && runtime == RuntimeState::Unknown) ++unknown_count;
        if (mask.witness && witness == WitnessState::Unknown) ++unknown_count;
        if (mask.health && health == HealthState::Unknown) ++unknown_count;
        if (unknown_count >= 2) {
            const Scalar unc = 0.02f + static_cast<Scalar>(unknown_count) * 0.04f;
            return {Choice::Unknown, Noul::Multiple, 0.15f, unc};
        }

        // 3. Degraded (Operational degradation)
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
            "{{\"id\":{},\"declared\":{},\"registered\":{},\"runtime\":\"{}\",\"witness\":\"{}\",\"freshness\":\"{}\",\"health\":\"{}\"}}",
            state_id(),
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
    std::size_t noul_target;  // Noul target class (locus diagnóstico)
    Scalar score_target;      // Learned viability score [0, 1]
    Scalar uncertainty_target;// Uncertainty variance target
    std::size_t state_id{0};  // Canonical state configuration ID
    std::string description;
};

struct OodSample {
    Vector x;
    OodCategory category;
    Choice expected_choice;
    std::string description;
};

struct DatasetSplit {
    std::vector<DataSample> train;
    std::vector<DataSample> val;
    std::vector<DataSample> test;      // Genuinely unseen state configurations
    std::vector<OodSample> ood;
};

class DatasetGenerator {
public:
    // Generate all 576 unique canonical state configurations in Ω
    static std::vector<StructuredState> generate_canonical_universe() {
        std::vector<StructuredState> universe;
        universe.reserve(576);

        for (int dec = 0; dec < 2; ++dec) {
            for (int reg = 0; reg < 2; ++reg) {
                for (int rt = 0; rt < 3; ++rt) {
                    for (int wt = 0; wt < 4; ++wt) {
                        for (int fr = 0; fr < 3; ++fr) {
                            for (int hl = 0; hl < 4; ++hl) {
                                StructuredState st{
                                    .declared = (dec == 1),
                                    .registered = (reg == 1),
                                    .runtime = static_cast<RuntimeState>(rt),
                                    .witness = static_cast<WitnessState>(wt),
                                    .freshness = static_cast<FreshnessState>(fr),
                                    .health = static_cast<HealthState>(hl)
                                };
                                universe.push_back(st);
                            }
                        }
                    }
                }
            }
        }
        return universe;
    }

    // Partition strictly by unique state configuration (zero state overlap across splits)
    static DatasetSplit generate_canonical_split(std::uint64_t seed = 42) {
        Random rng(seed);
        auto universe = generate_canonical_universe();
        rng.shuffle(universe);

        // Separate holdout novel combinations: e.g. when witness == Stale and runtime == Absent
        std::vector<StructuredState> in_dist_universe;
        std::vector<StructuredState> held_out_novel;

        for (const auto& st : universe) {
            if (st.witness == WitnessState::Stale && st.runtime == RuntimeState::Absent) {
                held_out_novel.push_back(st);
            } else {
                in_dist_universe.push_back(st);
            }
        }

        const std::size_t total = in_dist_universe.size();
        const std::size_t train_size = (total * 70) / 100;
        const std::size_t val_size = (total * 15) / 100;

        DatasetSplit split;

        auto make_sample = [](const StructuredState& st) -> DataSample {
            TripleJudgment j = st.evaluate_judgment();
            return {
                .x = st.encode(),
                .y = static_cast<std::size_t>(j.choice),
                .noul_target = static_cast<std::size_t>(j.noul),
                .score_target = j.score,
                .uncertainty_target = j.uncertainty,
                .state_id = st.state_id(),
                .description = st.to_json_str()
            };
        };

        for (std::size_t i = 0; i < total; ++i) {
            const auto& st = in_dist_universe[i];
            if (i < train_size) {
                split.train.push_back(make_sample(st));
            } else if (i < train_size + val_size) {
                split.val.push_back(make_sample(st));
            } else {
                split.test.push_back(make_sample(st));
            }
        }

        // Categorized OOD Evaluation Suite
        // 1. NO_SIGNAL (All zeros)
        split.ood.push_back({
            .x = Vector(24, 0.0f),
            .category = OodCategory::NoSignal,
            .expected_choice = Choice::Unknown,
            .description = "NO_SIGNAL: All zeros (zero evidence, expect high entropy H(P))"
        });

        // 2. UNIFORM_DISPERSION
        Vector uniform_v(24, 1.0f / 24.0f);
        split.ood.push_back({
            .x = uniform_v,
            .category = OodCategory::UniformDispersion,
            .expected_choice = Choice::Unknown,
            .description = "UNIFORM_DISPERSION: Flat uniform probability vector"
        });

        // 3. CONTRADICTORY_EVIDENCE (Simultaneous conflicting signals where P(INCONSISTENT)=1.0 is the right judgment)
        Vector multihot_v(24, 0.0f);
        multihot_v[0] = 1.0f; multihot_v[1] = 1.0f; // declared true AND false
        multihot_v[4] = 1.0f; multihot_v[5] = 1.0f; // running AND absent
        multihot_v[7] = 1.0f; multihot_v[8] = 1.0f; // valid AND invalid
        for (std::size_t k = 18; k < 24; ++k) multihot_v[k] = 1.0f;
        split.ood.push_back({
            .x = multihot_v,
            .category = OodCategory::ContradictoryEvidence,
            .expected_choice = Choice::Inconsistent,
            .description = "CONTRADICTORY_EVIDENCE: Multihot direct contradiction (expect P(INCONSISTENT) -> 1.0)"
        });

        // 4. NOVEL_COMBINATION (Held-out structural combinations)
        for (std::size_t k = 0; k < std::min(std::size_t(5), held_out_novel.size()); ++k) {
            split.ood.push_back({
                .x = held_out_novel[k].encode(),
                .category = OodCategory::NovelCombination,
                .expected_choice = held_out_novel[k].ground_truth(),
                .description = std::format("NOVEL_COMBINATION: Entirely held-out structural family (state_id={})", held_out_novel[k].state_id())
            });
        }

        // 5. CORRUPTED_INPUT (Continuous corrupted noise)
        for (int k = 0; k < 3; ++k) {
            Vector corrupt_v(24);
            for (auto& val : corrupt_v) val = std::abs(rng.normal(0.5f, 0.3f));
            split.ood.push_back({
                .x = corrupt_v,
                .category = OodCategory::CorruptedInput,
                .expected_choice = Choice::Unknown,
                .description = std::format("CORRUPTED_INPUT: Gaussian out-of-domain noise #{}", k + 1)
            });
        }

        return split;
    }

    // Mathematical verification that splits share zero state configurations
    static void verify_disjoint_partitions(const DatasetSplit& split) {
        std::unordered_set<std::size_t> train_ids;
        for (const auto& s : split.train) train_ids.insert(s.state_id);

        for (const auto& s : split.val) {
            if (train_ids.contains(s.state_id)) {
                throw std::runtime_error(std::format("Data leakage detected: val state_id {} exists in train", s.state_id));
            }
        }

        for (const auto& s : split.test) {
            if (train_ids.contains(s.state_id)) {
                throw std::runtime_error(std::format("Data leakage detected: test state_id {} exists in train", s.state_id));
            }
        }
    }

    // EXP-001 & EXP-002 compatibility generator wrapping canonical split
    static DatasetSplit generate_exp001([[maybe_unused]] std::size_t _unused = 250, std::uint64_t seed = 42) {
        return generate_canonical_split(seed);
    }

    // EXP-003 generator with explicit missingness on unique states
    static DatasetSplit generate_exp003([[maybe_unused]] std::size_t _unused = 300, std::uint64_t seed = 42) {
        auto base_split = generate_canonical_split(seed);
        Random rng(seed + 100);

        auto apply_missingness = [&](std::vector<DataSample>& samples) {
            for (auto& s : samples) {
                PresenceMask mask;
                const float rate = rng.uniform(0.0f, 0.35f);
                mask.declared = (rng.uniform() > rate);
                mask.registered = (rng.uniform() > rate);
                mask.runtime = (rng.uniform() > rate);
                mask.witness = (rng.uniform() > rate);
                mask.freshness = (rng.uniform() > rate);
                mask.health = (rng.uniform() > rate);

                // Recover underlying state
                auto universe = generate_canonical_universe();
                for (const auto& st : universe) {
                    if (st.state_id() == s.state_id) {
                        TripleJudgment j = st.evaluate_judgment_with_mask(mask);
                        s.x = st.encode_with_mask(mask);
                        s.y = static_cast<std::size_t>(j.choice);
                        s.noul_target = static_cast<std::size_t>(j.noul);
                        s.score_target = j.score;
                        s.uncertainty_target = j.uncertainty;
                        break;
                    }
                }
            }
        };

        apply_missingness(base_split.train);
        apply_missingness(base_split.val);
        apply_missingness(base_split.test);

        return base_split;
    }
};

enum class TextualStyle : std::size_t {
    NaturalProse = 0,
    TelemetryLog = 1,
    DiagnosticReport = 2,
    CompactKeyValue = 3
};

class TextualStateGenerator {
public:
    static std::string generate_text(const StructuredState& s, TextualStyle style) {
        std::string decl_str = s.declared ? "true" : "false";
        std::string reg_str  = s.registered ? "true" : "false";
        std::string run_str  = s.runtime == RuntimeState::Running ? "running" : (s.runtime == RuntimeState::Absent ? "absent" : "unknown");
        std::string wit_str  = s.witness == WitnessState::Valid ? "valid" : (s.witness == WitnessState::Invalid ? "invalid" : (s.witness == WitnessState::Stale ? "stale" : "unknown"));
        std::string fresh_str = s.freshness == FreshnessState::Fresh ? "fresh" : (s.freshness == FreshnessState::Aging ? "aging" : "expired");
        std::string hlth_str = s.health == HealthState::Healthy ? "healthy" : (s.health == HealthState::Degraded ? "degraded" : (s.health == HealthState::Failing ? "failing" : "unknown"));

        switch (style) {
            case TextualStyle::NaturalProse:
                return std::format(
                    "declaration is {} and registration is {}. runtime process is {} and witness is {}. freshness is {} and health is {}.",
                    decl_str, reg_str, run_str, wit_str, fresh_str, hlth_str
                );
            case TextualStyle::TelemetryLog:
                return std::format(
                    "[status] decl={} | reg={} | run={} | wit={} | fresh={} | hlth={}",
                    decl_str, reg_str, run_str, wit_str, fresh_str, hlth_str
                );
            case TextualStyle::DiagnosticReport:
                return std::format(
                    "service declaration: {}; discovery: {}; execution: {}; verification: {}; telemetry: {}; metric: {}",
                    decl_str, reg_str, run_str, wit_str, fresh_str, hlth_str
                );
            case TextualStyle::CompactKeyValue:
                return std::format(
                    "decl:{} reg:{} run:{} wit:{} fresh:{} hlth:{}",
                    decl_str, reg_str, run_str, wit_str, fresh_str, hlth_str
                );
        }
        return "";
    }
};

} // namespace tso
