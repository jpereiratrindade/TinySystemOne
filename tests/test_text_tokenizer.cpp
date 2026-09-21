#include "tso/text_tokenizer.hpp"
#include "tso/dataset.hpp"
#include <iostream>
#include <cassert>
#include <string>
#include <vector>

using namespace tso;

void test_text_tokenizer_basics() {
    std::cout << "[Test 1] Testing TextTokenizer vocabulary and basic splitting...\n";
    TextTokenizer tokenizer;

    assert(tokenizer.vocab_size() > 50);
    assert(tokenizer.token_to_string(tokenizer.cls_id()) == "[cls]");
    assert(tokenizer.token_to_string(tokenizer.sep_id()) == "[sep]");
    assert(tokenizer.token_to_string(tokenizer.pad_id()) == "[pad]");

    std::string text = "declaration is true and runtime process is running.";
    auto tokens = tokenizer.tokenize(text, 32);

    assert(tokens.size() == 32);
    assert(tokens[0] == static_cast<TokenId>(tokenizer.cls_id()));
    assert(tokens[1] == static_cast<TokenId>(tokenizer.string_to_token("declaration")));
    assert(tokens[2] == static_cast<TokenId>(tokenizer.string_to_token("is")));
    assert(tokens[3] == static_cast<TokenId>(tokenizer.string_to_token("true")));

    std::cout << "  TextTokenizer Basics PASSED (decoded: " << tokenizer.decode(tokens, true) << ")\n";
}

void test_text_tokenizer_case_and_punct() {
    std::cout << "[Test 2] Testing case insensitivity and punctuation handling...\n";
    TextTokenizer tokenizer;

    std::string text1 = "DECL:true | REG:false";
    std::string text2 = "decl: true | reg: false";

    auto tok1 = tokenizer.tokenize(text1, 16);
    auto tok2 = tokenizer.tokenize(text2, 16);

    assert(tok1.size() == 16);
    assert(tok2.size() == 16);

    // Both should decode identically when ignoring whitespace differences
    assert(tokenizer.decode(tok1, true) == tokenizer.decode(tok2, true));
    std::cout << "  Case & Punctuation Invariance PASSED\n";
}

void test_textual_state_generator() {
    std::cout << "[Test 3] Testing TextualStateGenerator templates...\n";
    StructuredState s{
        .declared = true,
        .registered = true,
        .runtime = RuntimeState::Running,
        .witness = WitnessState::Valid,
        .freshness = FreshnessState::Fresh,
        .health = HealthState::Healthy
    };

    std::string natural = TextualStateGenerator::generate_text(s, TextualStyle::NaturalProse);
    std::string log     = TextualStateGenerator::generate_text(s, TextualStyle::TelemetryLog);
    std::string report  = TextualStateGenerator::generate_text(s, TextualStyle::DiagnosticReport);
    std::string compact = TextualStateGenerator::generate_text(s, TextualStyle::CompactKeyValue);

    assert(!natural.empty());
    assert(!log.empty());
    assert(!report.empty());
    assert(!compact.empty());

    TextTokenizer tokenizer;
    auto tok_nat = tokenizer.tokenize(natural, 32);
    auto tok_log = tokenizer.tokenize(log, 32);

    assert(tok_nat.size() == 32);
    assert(tok_log.size() == 32);

    std::cout << "  TextualStateGenerator PASSED\n";
}

int main() {
    std::cout << "=== Running TinySystemOne Text Tokenizer Tests ===\n";
    test_text_tokenizer_basics();
    test_text_tokenizer_case_and_punct();
    test_textual_state_generator();
    std::cout << "=== ALL TEXT TOKENIZER TESTS PASSED ===\n";
    return 0;
}
