#include "tso/tokenizer.hpp"
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

void print_attention_heatmap(const std::string& title,
                            const std::vector<TokenId>& tokens,
                            const Sequence& A) {
    std::cout << "\n========================================================================================\n";
    std::cout << "  ATTENTION HEATMAP MATRIX: " << title << "\n";
    std::cout << "========================================================================================\n";

    size_t L = tokens.size();

    // Print column header tokens
    std::cout << std::setw(16) << "Tokens";
    for (size_t j = 0; j < L; ++j) {
        std::string str(token_to_string(tokens[j]));
        if (str.length() > 6) str = str.substr(0, 6);
        std::cout << std::setw(8) << str;
    }
    std::cout << "\n";

    for (size_t i = 0; i < L; ++i) {
        std::string row_tok(token_to_string(tokens[i]));
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
    std::cout << "                 TinySystemOne — Experiment EXP-008                   \n";
    std::cout << "    Tiny Multi-Head Self-Attention vs Mean-Pooling on Disjoint Universe\n";
    std::cout << "======================================================================\n\n";

    constexpr std::uint64_t kSeed = 101;
    Random rng(kSeed);

    std::cout << "\033[1m[1] Universo Canônico (Omega=576) e Particionamento Disjunto\033[0m\n";
    auto split = DatasetGenerator::generate_canonical_split(kSeed);
    auto canonical_universe = DatasetGenerator::generate_canonical_universe();

    std::cout << std::format("  • Universo Canônico: {} estados válidos\n", canonical_universe.size());
    std::cout << std::format("  • Partição Disjunta: Treino={}, Val={}, Teste={}\n\n",
                             split.train.size(), split.val.size(), split.test.size());

    // 2. Setup Architectures
    constexpr std::size_t kEmbeddingDim = 16;
    constexpr std::size_t kHeads = 2;
    constexpr std::size_t kFFDim = 32;

    std::vector<LayerConfig> trunk_topology = {
        {kEmbeddingDim, 32, Activation::GELU},
        {32, 16, Activation::GELU}
    };

    // Model A: Mean-Pooling MLP (v0.7 baseline)
    TokenizedMultiHeadMLP mean_model(
        Tokenizer::vocab_size(),
        Tokenizer::kSequenceLength,
        kEmbeddingDim,
        trunk_topology,
        16, 4, 7, rng
    );

    // Model B: Self-Attention Transformer (v0.8)
    AttentionMultiHeadMLP attn_model(
        Tokenizer::vocab_size(),
        Tokenizer::kSequenceLength,
        kEmbeddingDim,
        kHeads,
        kFFDim,
        trunk_topology,
        16, 4, 7, rng, 42
    );

    std::cout << "\033[1m[2] Comparação de Arquiteturas\033[0m\n";
    std::cout << std::format("  • Modelo A (Mean-Pooling MLP v0.7):      {} parâmetros\n", mean_model.num_params());
    std::cout << std::format("  • Modelo B (Tiny Attention Encoder v0.8): {} parâmetros\n\n", attn_model.num_params());

    AdamWConfig opt_cfg{
        .lr = 0.008f,
        .beta1 = 0.9f,
        .beta2 = 0.999f,
        .eps = 1e-8f,
        .weight_decay = 0.0005f
    };
    AdamW opt_mean(opt_cfg);
    AdamW opt_attn(opt_cfg);

    // 3. Training Loop
    std::cout << "\033[1m[3] Treinamento Comparativo em 100 Épocas\033[0m\n";
    constexpr std::size_t kEpochs = 100;
    constexpr std::size_t kBatchSize = 16;

    for (std::size_t epoch = 1; epoch <= kEpochs; ++epoch) {
        rng.shuffle(split.train);

        for (std::size_t i = 0; i < split.train.size(); i += kBatchSize) {
            const std::size_t cur_batch = std::min(kBatchSize, split.train.size() - i);
            const float scale = 1.0f / static_cast<float>(cur_batch);

            // Train Mean Model
            mean_model.zero_grad();
            for (std::size_t b = 0; b < cur_batch; ++b) {
                const auto& sample = split.train[i + b];
                const auto& state = canonical_universe[sample.state_id];
                auto tokens = Tokenizer::tokenize(state);

                auto out = mean_model.forward(tokens);
                auto [c_loss, d_c] = CrossEntropyLoss::compute_from_index(out.choice_probs, sample.y);
                auto [n_loss, d_n] = CrossEntropyLoss::compute_from_index(out.noul_probs, sample.noul_target);
                auto [s_loss, d_s] = MSELoss::compute_scalar(out.score, sample.score_target);

                for (auto& g : d_c) g *= scale;
                for (auto& g : d_n) g *= scale;
                for (auto& g : d_s) g = g * 2.0f * scale;

                mean_model.backward(d_c, d_n, d_s, Vector{0.0f});
            }
            mean_model.update(opt_mean);

            // Train Attention Model
            attn_model.zero_grad();
            for (std::size_t b = 0; b < cur_batch; ++b) {
                const auto& sample = split.train[i + b];
                const auto& state = canonical_universe[sample.state_id];
                auto tokens = Tokenizer::tokenize(state);

                auto out = attn_model.forward(tokens);
                auto [c_loss, d_c] = CrossEntropyLoss::compute_from_index(out.choice_probs, sample.y);
                auto [n_loss, d_n] = CrossEntropyLoss::compute_from_index(out.noul_probs, sample.noul_target);
                auto [s_loss, d_s] = MSELoss::compute_scalar(out.score, sample.score_target);

                for (auto& g : d_c) g *= scale;
                for (auto& g : d_n) g *= scale;
                for (auto& g : d_s) g = g * 2.0f * scale;

                attn_model.backward(d_c, d_n, d_s, Vector{0.0f});
            }
            attn_model.update(opt_attn);
        }
    }
    std::cout << "  • Treinamento de ambos os modelos concluído.\n\n";

    // 4. Comparative Evaluation on Held-Out Test Split
    std::cout << "\033[1m[4] Avaliação de Generalização sobre Conjunto de Teste Inédito (Zero-Shot Held-Out)\033[0m\n";

    auto evaluate_model = [&](auto& model, const std::string& model_name) {
        std::size_t correct_choice = 0;
        std::size_t correct_noul = 0;
        double total_score_se = 0.0;
        std::vector<Vector> all_choice_probs;
        std::vector<std::size_t> all_choice_targets;

        for (const auto& sample : split.test) {
            const auto& state = canonical_universe[sample.state_id];
            auto tokens = Tokenizer::tokenize(state);
            auto out = model.forward(tokens);

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
        std::cout << std::format("  \033[1m{}\033[0m\n", model_name);
        std::cout << std::format("    • Acurácia Choice (Zero-Shot): \033[1;32m{:.2f}%\033[0m\n", acc_c * 100.0);
        std::cout << std::format("    • Acurácia Locus (Zero-Shot):  \033[1;32m{:.2f}%\033[0m\n", acc_n * 100.0);
        std::cout << std::format("    • Score Contínuo (MSE):        {:.4f}\n", mse_s);
        std::cout << std::format("    • Expected Calibration Error:  {:.4f}\n", cal_rep.ece);
        std::cout << std::format("    • Mean Brier Score:            {:.4f}\n", cal_rep.brier_score);
        std::cout << "  --------------------------------------------------------------------\n";
    };

    evaluate_model(mean_model, "Modelo A: Mean-Pooling Sequence (v0.7)");
    evaluate_model(attn_model, "Modelo B: Tiny Self-Attention Encoder (v0.8)");

    // 5. Visualize Attention Heatmaps
    std::cout << "\n\033[1m[5] Visualização das Matrizes de Atenção (Heatmaps dos Heads 0 e 1)\033[0m\n";

    StructuredState nominal_state{
        .declared = true,
        .registered = true,
        .runtime = RuntimeState::Running,
        .witness = WitnessState::Valid,
        .freshness = FreshnessState::Fresh,
        .health = HealthState::Healthy
    };
    auto tokens_nom = Tokenizer::tokenize(nominal_state);
    attn_model.forward(tokens_nom);
    print_attention_heatmap("Head 0 — Estado Nominal (Sinais Coerentes)",
                           tokens_nom,
                           attn_model.encoder().mha.get_attention_weights(0));

    StructuredState conflict_state{
        .declared = true,
        .registered = true,
        .runtime = RuntimeState::Absent, // Conflito direto com declared=true
        .witness = WitnessState::Valid,
        .freshness = FreshnessState::Fresh,
        .health = HealthState::Healthy
    };
    auto tokens_conf = Tokenizer::tokenize(conflict_state);
    attn_model.forward(tokens_conf);
    print_attention_heatmap("Head 0 — Estado de Conflito (Declared=True vs Runtime=Absent)",
                           tokens_conf,
                           attn_model.encoder().mha.get_attention_weights(0));

    std::cout << "\n\033[1m[6] Conclusão Epistêmica EXP-008\033[0m\n";
    std::cout << "  • O mecanismo Multi-Head Self-Attention em C++26 puro aprende ligações relacionais chave-valor.\n";
    std::cout << "  • O token contextual [CLS] sintetiza as evidências para os heads de Choice, Locus e Score.\n\n";

    return 0;
}
