#include "tso/classifier.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <filesystem>

#define QA_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "\033[1;31m[TEST_QUESTION_CONDITIONING FAILED]\033[0m " << msg \
                      << " (" #cond ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(1); \
        } \
    } while (0)

void test_qa_tokenization() {
    std::cout << "[1] Testing Dual-Segment QA Tokenization...\n";
    tso::TextTokenizer tokenizer;

    std::string state_text = "declaration is true and registration is true.";
    std::string question_text = "question: is witness valid?";

    auto tokens = tokenizer.tokenize_qa(state_text, question_text, 48);
    QA_ASSERT(tokens.size() == 48, "Token sequence must match max length 48");
    QA_ASSERT(tokens[0] == static_cast<tso::TokenId>(tokenizer.cls_id()), "First token must be [CLS]");

    // Check that [SEP] occurs twice
    std::size_t sep_count = 0;
    for (auto tok : tokens) {
        if (tok == static_cast<tso::TokenId>(tokenizer.sep_id())) {
            ++sep_count;
        }
    }
    QA_ASSERT(sep_count == 2, "Must contain exactly two [SEP] tokens separating state and question");
    std::cout << "  ✓ Dual-segment QA tokenization verified!\n";
}

void test_dynamic_question_answering(const tso::SystemOneTextClassifier& classifier) {
    std::cout << "[2] Testing Dynamic Question Answering on Identical State Text...\n";

    // 1. Nominal State
    std::string nom_state = "declaration is true and registration is true. runtime process is running and witness is valid. freshness is fresh and health is healthy.";

    auto r_choice = classifier.ask(nom_state, "question: what is the overall system state?");
    QA_ASSERT(r_choice.choice == tso::Choice::Nominal, "Overall question on nominal state must yield Choice::Nominal");

    auto r_wit_valid = classifier.ask(nom_state, "question: is witness valid?");
    QA_ASSERT(r_wit_valid.noul_truth_probability >= 0.50f, "Witness valid query must have high Noul truth probability");

    // 2. Degraded Witness State
    std::string deg_state = "declaration is true and registration is true. runtime process is running and witness is invalid. freshness is fresh and health is healthy.";

    auto r_wit_invalid = classifier.ask(deg_state, "question: is witness valid?");
    QA_ASSERT(r_wit_invalid.noul_truth_probability <= 0.50f, "Witness invalid query must have low Noul truth probability");
    QA_ASSERT(r_wit_valid.noul_truth_probability > r_wit_invalid.noul_truth_probability, "Valid witness must score higher than invalid witness");

    std::cout << std::format("  • Nominal: P(witness_valid)={:.2f}%, Overall={}\n",
                             r_wit_valid.noul_truth_probability * 100.0f, tso::to_string(r_choice.choice));
    std::cout << std::format("  • Degraded: P(witness_valid)={:.2f}%\n",
                             r_wit_invalid.noul_truth_probability * 100.0f);
    std::cout << "  ✓ Dynamic Question Conditioning verified on varied questions!\n";
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "      TinySystemOne — Test Question-Conditioned Judgment (v2.0.0)     \n";
    std::cout << "======================================================================\n\n";

    test_qa_tokenization();

    auto split = tso::DatasetGenerator::generate_canonical_split(101);
    tso::TextClassifierTrainingConfig cfg{
        .epochs = 70,
        .batch_size = 16,
        .learning_rate = 0.006f,
        .weight_decay = 0.0005f,
        .ood_percentile = 0.98f,
        .seed = 101
    };

    std::cout << "• Training question-conditioned text transformer (70 epochs)...\n";
    auto classifier = tso::SystemOneTextClassifier::train_and_calibrate(split, cfg);

    test_dynamic_question_answering(classifier);

    std::cout << "\n\033[1;32m[ALL QUESTION CONDITIONING TESTS PASSED SUCCESSFULLY]\033[0m\n";
    return 0;
}
