#include "tso/tokenizer.hpp"
#include "tso/embedding.hpp"
#include "tso/model.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

#define TOKEN_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "\033[1;31m[TEST_TOKENIZER FAILED]\033[0m " << msg \
                      << " (" #cond ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(1); \
        } \
    } while (0)

void test_tokenization_and_decoding() {
    std::cout << "[1] Testing Tokenization and Sequence Reconstruction...\n";

    tso::StructuredState state{
        .declared = true,
        .registered = true,
        .runtime = tso::RuntimeState::Running,
        .witness = tso::WitnessState::Valid,
        .freshness = tso::FreshnessState::Fresh,
        .health = tso::HealthState::Healthy
    };

    auto tokens = tso::Tokenizer::tokenize(state);
    TOKEN_ASSERT(tokens.size() == tso::Tokenizer::kSequenceLength, "Sequence length must be 14");
    TOKEN_ASSERT(tokens[0] == tso::TokenId::Cls, "First token must be [CLS]");
    TOKEN_ASSERT(tokens[1] == tso::TokenId::KeyDeclared, "Second token must be KEY:declared");
    TOKEN_ASSERT(tokens[2] == tso::TokenId::ValTrue, "Third token must be VAL:true");
    TOKEN_ASSERT(tokens[13] == tso::TokenId::Sep, "Last token must be [SEP]");

    std::string decoded = tso::Tokenizer::decode(tokens);
    TOKEN_ASSERT(!decoded.empty(), "Decoded string must not be empty");
    std::cout << "  • Decoded: " << decoded << "\n";
    std::cout << "  ✓ Tokenization and decoding verified!\n";
}

void test_mask_tokenization() {
    std::cout << "[2] Testing Mask Token Insertion under Sensor Dropout...\n";

    tso::StructuredState state{
        .declared = true,
        .registered = true,
        .runtime = tso::RuntimeState::Running,
        .witness = tso::WitnessState::Valid,
        .freshness = tso::FreshnessState::Fresh,
        .health = tso::HealthState::Healthy
    };

    tso::PresenceMask mask{
        .declared = true,
        .registered = false, // Masked
        .runtime = true,
        .witness = false,    // Masked
        .freshness = true,
        .health = false      // Masked
    };

    auto tokens = tso::Tokenizer::tokenize(state, mask);
    TOKEN_ASSERT(tokens[4] == tso::TokenId::Mask, "Registered value must be [MASK]");
    TOKEN_ASSERT(tokens[8] == tso::TokenId::Mask, "Witness value must be [MASK]");
    TOKEN_ASSERT(tokens[12] == tso::TokenId::Mask, "Health value must be [MASK]");

    std::string decoded = tso::Tokenizer::decode(tokens);
    std::cout << "  • Masked Sequence: " << decoded << "\n";
    std::cout << "  ✓ Mask tokenization verified!\n";
}

void test_embedding_table_and_backprop() {
    std::cout << "[3] Testing Embedding Layer Forward, Backward & Optimizer Step...\n";

    tso::Random rng(42);
    constexpr std::size_t kVocab = tso::Tokenizer::vocab_size();
    constexpr std::size_t kSeqLen = tso::Tokenizer::kSequenceLength;
    constexpr std::size_t kDim = 16;

    tso::Embedding emb(kVocab, kSeqLen, kDim, rng);
    TOKEN_ASSERT(emb.num_params() == (kVocab * kDim + kSeqLen * kDim), "Param count must match table dimensions");

    tso::StructuredState state{};
    auto tokens = tso::Tokenizer::tokenize(state);

    // Forward
    tso::Vector pooled = emb.forward_mean_pooled(tokens);
    TOKEN_ASSERT(pooled.size() == kDim, "Pooled vector size must match embedding_dim");

    // Backward & update
    emb.zero_grad();
    tso::Vector grad_out(kDim, 0.5f);
    emb.backward_mean_pooled(grad_out);

    tso::AdamWConfig opt_cfg{.lr = 0.01f, .beta1 = 0.9f, .beta2 = 0.999f, .eps = 1e-8f, .weight_decay = 0.0f};
    tso::AdamW optimizer(opt_cfg);
    emb.update(optimizer);

    // Forward again to confirm weights changed
    tso::Vector pooled_after = emb.forward_mean_pooled(tokens);
    float diff = 0.0f;
    for (std::size_t i = 0; i < kDim; ++i) diff += std::abs(pooled[i] - pooled_after[i]);
    TOKEN_ASSERT(diff > 1e-4f, "Embedding forward output must reflect optimizer update");

    std::cout << "  ✓ Embedding table forward/backward/update verified!\n";
}

void test_tokenized_multi_head_model() {
    std::cout << "[4] Testing End-to-End TokenizedMultiHeadMLP...\n";

    tso::Random rng(123);
    std::vector<tso::LayerConfig> trunk = {
        {16, 24, tso::Activation::GELU},
        {24, 16, tso::Activation::GELU}
    };

    tso::TokenizedMultiHeadMLP model(
        tso::Tokenizer::vocab_size(),
        tso::Tokenizer::kSequenceLength,
        16, // embedding dim
        trunk,
        16, // latent dim
        4,  // choice dim
        7,  // noul dim
        rng
    );

    tso::StructuredState state{};
    auto tokens = tso::Tokenizer::tokenize(state);

    auto out = model.forward(tokens);
    TOKEN_ASSERT(out.choice_probs.size() == 4, "Choice probs size must be 4");
    TOKEN_ASSERT(out.noul_probs.size() == 7, "Noul probs size must be 7");

    model.zero_grad();
    model.backward(tso::Vector(4, 0.1f), tso::Vector(7, 0.1f), tso::Vector{0.1f}, tso::Vector{0.1f});

    tso::AdamW optimizer(tso::AdamWConfig{.lr = 0.005f, .beta1 = 0.9f, .beta2 = 0.999f, .eps = 1e-8f, .weight_decay = 0.0001f});
    model.update(optimizer);

    std::cout << "  ✓ End-to-end TokenizedMultiHeadMLP forward & backward verified!\n";
}

int main() {
    std::cout << "=== TinySystemOne Tokenizer & Embedding Test Suite ===\n";
    test_tokenization_and_decoding();
    test_mask_tokenization();
    test_embedding_table_and_backprop();
    test_tokenized_multi_head_model();
    std::cout << "\033[1;32mAll Tokenizer & Embedding tests passed successfully!\033[0m\n";
    return 0;
}
