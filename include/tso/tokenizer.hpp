#pragma once

#include "dataset.hpp"
#include <string>
#include <string_view>
#include <vector>
#include <array>
#include <format>
#include <stdexcept>

namespace tso {

enum class TokenId : std::size_t {
    // Special tokens
    Pad = 0,
    Unk = 1,
    Cls = 2,
    Sep = 3,
    Mask = 4,

    // Sensor Keys
    KeyDeclared = 5,
    KeyRegistered = 6,
    KeyRuntime = 7,
    KeyWitness = 8,
    KeyFreshness = 9,
    KeyHealth = 10,

    // Sensor Values
    ValTrue = 11,
    ValFalse = 12,
    ValRunning = 13,
    ValAbsent = 14,
    ValUnknown = 15,
    ValValid = 16,
    ValInvalid = 17,
    ValStale = 18,
    ValFresh = 19,
    ValAging = 20,
    ValExpired = 21,
    ValHealthy = 22,
    ValDegraded = 23,
    ValFailing = 24
};

inline constexpr std::size_t kVocabSize = 25;

inline std::string_view token_to_string(TokenId tok) {
    switch (tok) {
        case TokenId::Pad: return "[PAD]";
        case TokenId::Unk: return "[UNK]";
        case TokenId::Cls: return "[CLS]";
        case TokenId::Sep: return "[SEP]";
        case TokenId::Mask: return "[MASK]";
        case TokenId::KeyDeclared: return "KEY:declared";
        case TokenId::KeyRegistered: return "KEY:registered";
        case TokenId::KeyRuntime: return "KEY:runtime";
        case TokenId::KeyWitness: return "KEY:witness";
        case TokenId::KeyFreshness: return "KEY:freshness";
        case TokenId::KeyHealth: return "KEY:health";
        case TokenId::ValTrue: return "VAL:true";
        case TokenId::ValFalse: return "VAL:false";
        case TokenId::ValRunning: return "VAL:running";
        case TokenId::ValAbsent: return "VAL:absent";
        case TokenId::ValUnknown: return "VAL:unknown";
        case TokenId::ValValid: return "VAL:valid";
        case TokenId::ValInvalid: return "VAL:invalid";
        case TokenId::ValStale: return "VAL:stale";
        case TokenId::ValFresh: return "VAL:fresh";
        case TokenId::ValAging: return "VAL:aging";
        case TokenId::ValExpired: return "VAL:expired";
        case TokenId::ValHealthy: return "VAL:healthy";
        case TokenId::ValDegraded: return "VAL:degraded";
        case TokenId::ValFailing: return "VAL:failing";
    }
    return "[UNKNOWN_TOKEN]";
}

class Tokenizer {
public:
    // Tokenizes a StructuredState with PresenceMask into a fixed or sequence representation
    // Format: [CLS] KEY_DECL VAL KEY_REG VAL KEY_RUN VAL KEY_WIT VAL KEY_FRESH VAL KEY_HEALTH VAL [SEP]
    // Sequence length: 1 + (6 * 2) + 1 = 14 tokens
    static constexpr std::size_t kSequenceLength = 14;

    static std::vector<TokenId> tokenize(const StructuredState& state, const PresenceMask& mask = {}) {
        std::vector<TokenId> tokens;
        tokens.reserve(kSequenceLength);

        // 0. [CLS]
        tokens.push_back(TokenId::Cls);

        // 1. Declared
        tokens.push_back(TokenId::KeyDeclared);
        if (mask.declared) {
            tokens.push_back(state.declared ? TokenId::ValTrue : TokenId::ValFalse);
        } else {
            tokens.push_back(TokenId::Mask);
        }

        // 2. Registered
        tokens.push_back(TokenId::KeyRegistered);
        if (mask.registered) {
            tokens.push_back(state.registered ? TokenId::ValTrue : TokenId::ValFalse);
        } else {
            tokens.push_back(TokenId::Mask);
        }

        // 3. Runtime
        tokens.push_back(TokenId::KeyRuntime);
        if (mask.runtime) {
            switch (state.runtime) {
                case RuntimeState::Running: tokens.push_back(TokenId::ValRunning); break;
                case RuntimeState::Absent:  tokens.push_back(TokenId::ValAbsent); break;
                case RuntimeState::Unknown: tokens.push_back(TokenId::ValUnknown); break;
            }
        } else {
            tokens.push_back(TokenId::Mask);
        }

        // 4. Witness
        tokens.push_back(TokenId::KeyWitness);
        if (mask.witness) {
            switch (state.witness) {
                case WitnessState::Valid:   tokens.push_back(TokenId::ValValid); break;
                case WitnessState::Invalid: tokens.push_back(TokenId::ValInvalid); break;
                case WitnessState::Stale:   tokens.push_back(TokenId::ValStale); break;
                case WitnessState::Unknown: tokens.push_back(TokenId::ValUnknown); break;
            }
        } else {
            tokens.push_back(TokenId::Mask);
        }

        // 5. Freshness
        tokens.push_back(TokenId::KeyFreshness);
        if (mask.freshness) {
            switch (state.freshness) {
                case FreshnessState::Fresh:   tokens.push_back(TokenId::ValFresh); break;
                case FreshnessState::Aging:   tokens.push_back(TokenId::ValAging); break;
                case FreshnessState::Expired: tokens.push_back(TokenId::ValExpired); break;
            }
        } else {
            tokens.push_back(TokenId::Mask);
        }

        // 6. Health
        tokens.push_back(TokenId::KeyHealth);
        if (mask.health) {
            switch (state.health) {
                case HealthState::Healthy:  tokens.push_back(TokenId::ValHealthy); break;
                case HealthState::Degraded: tokens.push_back(TokenId::ValDegraded); break;
                case HealthState::Failing:  tokens.push_back(TokenId::ValFailing); break;
                case HealthState::Unknown:  tokens.push_back(TokenId::ValUnknown); break;
            }
        } else {
            tokens.push_back(TokenId::Mask);
        }

        // 7. [SEP]
        tokens.push_back(TokenId::Sep);

        return tokens;
    }

    static std::string decode(const std::vector<TokenId>& tokens) {
        std::string s;
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            if (i > 0) s += " ";
            s += token_to_string(tokens[i]);
        }
        return s;
    }

    [[nodiscard]] static constexpr std::size_t vocab_size() noexcept {
        return kVocabSize;
    }
};

} // namespace tso
