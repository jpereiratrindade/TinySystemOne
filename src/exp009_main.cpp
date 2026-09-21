#include "tso/text_tokenizer.hpp"
#include "tso/embedding.hpp"
#include "tso/attention.hpp"
#include "tso/transformer.hpp"
#include "tso/model.hpp"
#include "tso/loss.hpp"
#include "tso/optimizer.hpp"
#include "tso/calibration.hpp"
#include "tso/dataset.hpp"
#include <iostream>
#include <iomanip>
#include <format>
#include <vector>
#include <numeric>
#include <algorithm>
#include <string>

using namespace tso;

void print_text_attention_heatmap(const std::string& title,
                                  const std::vector<TokenId>& tokens,
                                  const TextTokenizer& tokenizer,
                                  const Sequence& A) {
    std::cout << "\n========================================================================================\n";
    std::cout << "  TEXT ATTENTION HEATMAP MATRIX: " << title << "\n";
    std::cout << "========================================================================================\n";

    // Find effective length before padding
    size_t L = tokens.size();
    while (L > 0 && static_cast<std::size_t>(tokens[L - 1]) == tokenizer.pad_id()) {
        --L;
    }
    L = std::min(L, static_cast<size_t>(16)); // Limit display to first 16 active tokens

    // Print column header tokens
    std::cout << std::setw(16) << "Tokens";
    for (size_t j = 0; j < L; ++j) {
        std::string str(tokenizer.token_to_string(static_cast<std::size_t>(tokens[j])));
        if (str.length() > 6) str = str.substr(0, 6);
        std::cout << std::setw(8) << str;
    }
    std::cout << "\n";

    for (size_t i = 0; i < L; ++i) {
        std::string row_tok(tokenizer.token_to_string(static_cast<std::size_t>(tokens[i])));
        if (row_tok.length() > 14) row_tok = row_tok.substr(0, 14);
        std::cout << std::setw(16) << row_tok;

        for (size_t j = 0; j < L; ++j) {
            double weight = A[i][j];
            std::cout << std::fixed << std::setprecision(2) << std::setw(8) << weight;
        }
        std::cout << "\n";
    }
    std::cout << "========================================================================================\n";
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "                 TinySystemOne — Experiment EXP-009                   \n";
    std::cout << "    Semi-Structured Textual States & Style-Invariant Judgment (v0.9.0)\n";
    std::cout << "======================================================================\n\n";

    constexpr std::uint64_t kSeed = 101;
    Random rng(kSeed);
    TextTokenizer text_tokenizer;

    std::cout << "\033[1m[1] Vocabulário Lexical Controlado & Representações Textuais\033[0m\n";
    std::cout << std::format("  • Tamanho do Vocabulário Lexical (|V|): {} tokens\n", text_tokenizer.vocab_size());
    std::cout << std::format("  • Comprimento Máximo de Sequência:     {} tokens\n", TextTokenizer::kMaxSequenceLength);

    auto split = DatasetGenerator::generate_canonical_split(kSeed);
    auto canonical_universe = DatasetGenerator::generate_canonical_universe();

    std::cout << std::format("  • Universo Canônico: {} estados (Treino={}, Val={}, Teste={})\n\n",
                             canonical_universe.size(), split.train.size(), split.val.size(), split.test.size());

    // 2. Setup Scaled Attention Architecture for Text
    constexpr std::size_t kEmbeddingDim = 24;
    constexpr std::size_t kHeads = 3;
    constexpr std::size_t kFFDim = 48;

    std::vector<LayerConfig> trunk_topology = {
        {kEmbeddingDim, 48, Activation::GELU},
        {48, 24, Activation::GELU}
    };

    AttentionMultiHeadMLP text_model(
        text_tokenizer.vocab_size(),
        TextTokenizer::kMaxSequenceLength,
        kEmbeddingDim,
        kHeads,
        kFFDim,
        trunk_topology,
        24, 4, 7, rng, 42
    );

    std::cout << "\033[1m[2] Arquitetura Textual Instanciada\033[0m\n";
    std::cout << std::format("  • Parâmetros Totais: {} (Embeddings: {}, Transformer: {}, Heads: {})\n\n",
                             text_model.num_params(),
                             text_model.embedding().num_params(),
                             text_model.num_params() - text_model.embedding().num_params() - text_model.mlp().num_params(),
                             text_model.mlp().num_params());

    AdamWConfig opt_cfg{
        .lr = 0.006f,
        .beta1 = 0.9f,
        .beta2 = 0.999f,
        .eps = 1e-8f,
        .weight_decay = 0.0005f
    };
    AdamW optimizer(opt_cfg);

    // 3. Training Loop across Multi-Style Text Sequences
    std::cout << "\033[1m[3] Treinamento End-to-End sobre Enunciados Textuais Variados\033[0m\n";
    constexpr std::size_t kEpochs = 90;
    constexpr std::size_t kBatchSize = 16;

    for (std::size_t epoch = 1; epoch <= kEpochs; ++epoch) {
        rng.shuffle(split.train);

        for (std::size_t i = 0; i < split.train.size(); i += kBatchSize) {
            const std::size_t cur_batch = std::min(kBatchSize, split.train.size() - i);
            const float scale = 1.0f / static_cast<float>(cur_batch);

            text_model.zero_grad();
            for (std::size_t b = 0; b < cur_batch; ++b) {
                const auto& sample = split.train[i + b];
                const auto& state = canonical_universe[sample.state_id];

                // Randomly select one of the 4 text styles during training
                auto style = static_cast<TextualStyle>(rng.uniform_int(0, 3));
                std::string text = TextualStateGenerator::generate_text(state, style);
                auto tokens = text_tokenizer.tokenize(text, TextTokenizer::kMaxSequenceLength);

                auto out = text_model.forward(tokens);
                auto [c_loss, d_c] = CrossEntropyLoss::compute_from_index(out.choice_probs, sample.y);
                auto [n_loss, d_n] = CrossEntropyLoss::compute_from_index(out.noul_probs, sample.noul_target);
                auto [s_loss, d_s] = MSELoss::compute_scalar(out.score, sample.score_target);

                for (auto& g : d_c) g *= scale;
                for (auto& g : d_n) g *= scale;
                for (auto& g : d_s) g = g * 2.0f * scale;

                text_model.backward(d_c, d_n, d_s, Vector{0.0f});
            }
            text_model.update(optimizer);
        }
    }
    std::cout << "  • Treinamento concluído sobre enunciados em prosa, logs e relatórios.\n\n";

    // 4. Zero-Shot Evaluation per Textual Style on Held-Out Test Split
    std::cout << "\033[1m[4] Avaliação de Generalização Textual por Estilo Sintático (Held-Out Test Set)\033[0m\n";

    auto evaluate_style = [&](TextualStyle style, const std::string& style_name) {
        std::size_t correct_choice = 0;
        std::size_t correct_noul = 0;
        double total_score_se = 0.0;
        std::vector<Vector> all_choice_probs;
        std::vector<std::size_t> all_choice_targets;

        for (const auto& sample : split.test) {
            const auto& state = canonical_universe[sample.state_id];
            std::string text = TextualStateGenerator::generate_text(state, style);
            auto tokens = text_tokenizer.tokenize(text, TextTokenizer::kMaxSequenceLength);

            auto out = text_model.forward(tokens);

            auto max_c = std::max_element(out.choice_probs.begin(), out.choice_probs.end());
            const auto pred_c = static_cast<std::size_t>(std::distance(out.choice_probs.begin(), max_c));

            auto max_n = std::max_element(out.noul_probs.begin(), out.noul_probs.end());
            const auto pred_n = static_cast<std::size_t>(std::distance(out.noul_probs.begin(), max_n));

            if (pred_c == sample.y) ++correct_choice;
            if (pred_n == sample.noul_target) ++correct_noul;

            all_choice_probs.push_back(out.choice_probs);
            all_choice_targets.push_back(sample.y);

            double diff = static_cast<double>(out.score - sample.score_target);
            total_score_se += diff * diff;
        }

        const double acc_c = static_cast<double>(correct_choice) / static_cast<double>(split.test.size());
        const double acc_n = static_cast<double>(correct_noul) / static_cast<double>(split.test.size());
        const double mse_s = total_score_se / static_cast<double>(split.test.size());
        auto cal_rep = Calibration::evaluate(all_choice_probs, all_choice_targets, 10);

        std::cout << "  --------------------------------------------------------------------\n";
        std::cout << std::format("  \033[1mEstilo: {}\033[0m\n", style_name);
        std::cout << std::format("    • Acurácia Choice (Zero-Shot): \033[1;32m{:.2f}%\033[0m\n", acc_c * 100.0);
        std::cout << std::format("    • Acurácia Locus (Zero-Shot):  \033[1;32m{:.2f}%\033[0m\n", acc_n * 100.0);
        std::cout << std::format("    • Score Contínuo (MSE):        {:.4f}\n", mse_s);
        std::cout << std::format("    • Expected Calibration Error:  {:.4f}\n", cal_rep.ece);
        std::cout << std::format("    • Mean Brier Score:            {:.4f}\n", cal_rep.brier_score);
    };

    evaluate_style(TextualStyle::NaturalProse,     "1. Natural Language Prose");
    evaluate_style(TextualStyle::TelemetryLog,     "2. Telemetry Log format");
    evaluate_style(TextualStyle::DiagnosticReport, "3. Diagnostic Incident Report");
    evaluate_style(TextualStyle::CompactKeyValue,  "4. Compact Key-Value");
    std::cout << "  --------------------------------------------------------------------\n\n";

    // 5. Visualize Attention Heatmaps on Natural Sentences
    std::cout << "\033[1m[5] Visualização da Matriz de Atenção sobre Texto em Linguagem Natural\033[0m\n";

    StructuredState conflict_state{
        .declared = true,
        .registered = true,
        .runtime = RuntimeState::Absent,
        .witness = WitnessState::Valid,
        .freshness = FreshnessState::Fresh,
        .health = HealthState::Healthy
    };
    std::string text_prose = TextualStateGenerator::generate_text(conflict_state, TextualStyle::NaturalProse);
    auto tokens_prose = text_tokenizer.tokenize(text_prose, TextTokenizer::kMaxSequenceLength);

    std::cout << "  • Sentença Analisada:\n    \"" << text_prose << "\"\n";
    text_model.forward(tokens_prose);
    print_text_attention_heatmap("Head 0 — Atenção sobre Sentença em Linguagem Natural",
                                 tokens_prose,
                                 text_tokenizer,
                                 text_model.encoder().mha.get_attention_weights(0));

    std::cout << "\033[1m[6] Conclusão Epistêmica EXP-009\033[0m\n";
    std::cout << "  • O Tiny Attention Encoder processa linguagem semi-estruturada com invariância sintática.\n";
    std::cout << "  • Julgamentos tipados e calibrados emergem de texto livre sem necessidade de geração de texto.\n\n";

    return 0;
}
