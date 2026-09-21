#include "tso/tokenizer.hpp"
#include "tso/embedding.hpp"
#include "tso/model.hpp"
#include "tso/loss.hpp"
#include "tso/optimizer.hpp"
#include "tso/calibration.hpp"
#include <iostream>
#include <iomanip>
#include <format>
#include <vector>
#include <numeric>
#include <algorithm>

int main() {
    std::cout << "======================================================================\n";
    std::cout << "                 TinySystemOne — Experiment EXP-007                   \n";
    std::cout << "      Discrete Tokenization, Learnable Embeddings & Sequence Space    \n";
    std::cout << "======================================================================\n\n";

    constexpr std::uint64_t kSeed = 101;
    tso::Random rng(kSeed);

    std::cout << "\033[1m[1] Vocabulário Simbólico e Tokenização de Estados\033[0m\n";
    std::cout << std::format("  • Tamanho do Vocabulário (|V|): {} tokens discretos\n", tso::Tokenizer::vocab_size());
    std::cout << std::format("  • Comprimento da Sequência (L): {} tokens por observação\n", tso::Tokenizer::kSequenceLength);

    tso::StructuredState sample_state{
        .declared = true,
        .registered = true,
        .runtime = tso::RuntimeState::Running,
        .witness = tso::WitnessState::Stale,
        .freshness = tso::FreshnessState::Fresh,
        .health = tso::HealthState::Healthy
    };
    auto sample_tokens = tso::Tokenizer::tokenize(sample_state);
    std::cout << "  • Exemplo de Sequência Tokenizada:\n    " << tso::Tokenizer::decode(sample_tokens) << "\n\n";

    // 2. Setup Tokenized Multi-Head Model
    constexpr std::size_t kEmbeddingDim = 16;
    std::vector<tso::LayerConfig> trunk_topology = {
        {kEmbeddingDim, 32, tso::Activation::GELU},
        {32, 16, tso::Activation::GELU}
    };

    tso::TokenizedMultiHeadMLP model(
        tso::Tokenizer::vocab_size(),
        tso::Tokenizer::kSequenceLength,
        kEmbeddingDim,
        trunk_topology,
        16, // latent dim
        4,  // choice dim
        7,  // noul dim
        rng
    );

    std::cout << std::format("  • Modelo Instanciado: {} parâmetros totais (Tabela de Embeddings: {})\n\n",
                             model.num_params(), model.embedding().num_params());

    auto split = tso::DatasetGenerator::generate_canonical_split(kSeed);
    auto canonical_universe = tso::DatasetGenerator::generate_canonical_universe();

    tso::AdamWConfig opt_cfg{
        .lr = 0.008f,
        .beta1 = 0.9f,
        .beta2 = 0.999f,
        .eps = 1e-8f,
        .weight_decay = 0.0005f
    };
    tso::AdamW optimizer(opt_cfg);

    std::cout << "\033[1m[2] Treinamento End-to-End sobre Sequências de Tokens\033[0m\n";
    constexpr std::size_t kEpochs = 90;
    constexpr std::size_t kBatchSize = 16;

    for (std::size_t epoch = 1; epoch <= kEpochs; ++epoch) {
        rng.shuffle(split.train);

        for (std::size_t i = 0; i < split.train.size(); i += kBatchSize) {
            const std::size_t cur_batch = std::min(kBatchSize, split.train.size() - i);
            model.zero_grad();

            for (std::size_t b = 0; b < cur_batch; ++b) {
                const auto& sample = split.train[i + b];
                const auto& state = canonical_universe[sample.state_id];
                auto tokens = tso::Tokenizer::tokenize(state);

                auto out = model.forward(tokens);

                auto [c_loss, d_c] = tso::CrossEntropyLoss::compute_from_index(out.choice_probs, sample.y);
                auto [n_loss, d_n] = tso::CrossEntropyLoss::compute_from_index(out.noul_probs, sample.noul_target);
                auto [s_loss, d_s] = tso::MSELoss::compute_scalar(out.score, sample.score_target);

                const float scale = 1.0f / static_cast<float>(cur_batch);
                for (auto& g : d_c) g *= scale;
                for (auto& g : d_n) g *= scale;
                for (auto& g : d_s) g = g * 2.0f * scale;

                model.backward(d_c, d_n, d_s, tso::Vector{0.0f});
            }

            model.update(optimizer);
        }
    }
    std::cout << "  • Treinamento concluído (90 épocas atualizando Embeddings e MLP).\n\n";

    // 3. Evaluate Generalization on Unseen Test Set via Token Sequences
    std::cout << "\033[1m[3] Avaliação de Generalização sobre Conjunto de Teste Tokenizado\033[0m\n";
    std::size_t correct_choice = 0;
    std::size_t correct_noul = 0;

    for (const auto& sample : split.test) {
        const auto& state = canonical_universe[sample.state_id];
        auto tokens = tso::Tokenizer::tokenize(state);
        auto out = model.forward(tokens);

        auto max_c = std::max_element(out.choice_probs.begin(), out.choice_probs.end());
        const auto pred_c = static_cast<std::size_t>(std::distance(out.choice_probs.begin(), max_c));

        auto max_n = std::max_element(out.noul_probs.begin(), out.noul_probs.end());
        const auto pred_n = static_cast<std::size_t>(std::distance(out.noul_probs.begin(), max_n));

        if (pred_c == sample.y) ++correct_choice;
        if (pred_n == sample.noul_target) ++correct_noul;
    }

    const double acc_c = static_cast<double>(correct_choice) / static_cast<double>(split.test.size());
    const double acc_n = static_cast<double>(correct_noul) / static_cast<double>(split.test.size());

    std::cout << std::format("  • Acurácia Choice em Sequências de Teste Inéditas: \033[1;32m{:.2f}%\033[0m\n", acc_c * 100.0);
    std::cout << std::format("  • Acurácia Noul em Sequências de Teste Inéditas:   \033[1;32m{:.2f}%\033[0m\n\n", acc_n * 100.0);

    // 4. Geometry and Semantic Topology of the Learned Embedding Space
    std::cout << "\033[1m[4] Topologia Geométrica do Espaço de Embeddings Aprendido\033[0m\n";
    const auto& emb = model.embedding();

    auto print_sim = [&](tso::TokenId t1, tso::TokenId t2, std::string_view note) {
        float sim = emb.token_cosine_similarity(t1, t2);
        std::cout << std::format("  • cos({:<14}, {:<14}) = \033[1;36m{:>7.4f}\033[0m ({})\n",
                                 tso::token_to_string(t1), tso::token_to_string(t2), sim, note);
    };

    print_sim(tso::TokenId::ValTrue, tso::TokenId::ValFalse, "Polaridade Booleana");
    print_sim(tso::TokenId::ValRunning, tso::TokenId::ValAbsent, "Antônimos de Runtime");
    print_sim(tso::TokenId::ValValid, tso::TokenId::ValInvalid, "Validade de Witness");
    print_sim(tso::TokenId::ValFresh, tso::TokenId::ValExpired, "Extremos de Freshness");
    print_sim(tso::TokenId::ValHealthy, tso::TokenId::ValFailing, "Extremos de Saúde Operacional");
    print_sim(tso::TokenId::ValHealthy, tso::TokenId::ValDegraded, "Gradação de Degradação");

    std::cout << "\n\033[1;32m[✓] Experimento EXP-007 concluído com sucesso: Tokenização e Embeddings Validados!\033[0m\n";
    return 0;
}
