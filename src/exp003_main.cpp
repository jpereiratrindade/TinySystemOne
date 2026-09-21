#include <tso/tensor.hpp>
#include <tso/random.hpp>
#include <tso/loss.hpp>
#include <tso/model.hpp>
#include <tso/optimizer.hpp>
#include <tso/calibration.hpp>
#include <tso/dataset.hpp>

#include <iostream>
#include <iomanip>
#include <format>
#include <vector>
#include <string>
#include <cmath>

void print_banner() {
    std::cout << "\033[1;36m";
    std::cout << "======================================================================\n";
    std::cout << "                 TinySystemOne — Experiment EXP-003                   \n";
    std::cout << "  Explicit Missing Values, Heteroscedastic Uncertainty & Dropout Test \n";
    std::cout << "======================================================================\033[0m\n\n";
}

int main() {
    print_banner();

    constexpr std::uint64_t kSeed = 42;
    tso::Random rng(kSeed);

    std::cout << "\033[1m[1] Configuração da Arquitetura 4-Head com Incerteza Epistêmica\033[0m\n";
    std::vector<tso::LayerConfig> trunk_topology = {
        {24, 32, tso::Activation::GELU},
        {32, 16, tso::Activation::GELU}
    };

    tso::MultiHeadMLP model(trunk_topology, 16, 4, 7, rng);
    std::cout << std::format("  • Entrada: 24 dims (18 atributos + 6 bits de máscara de presença)\n");
    std::cout << std::format("  • Trunk: 24 -> 32 (GELU) -> 16 (GELU)\n");
    std::cout << std::format("  • Choice Head:      16 -> 4 (Softmax)\n");
    std::cout << std::format("  • Noul Head:        16 -> 7 (Softmax)\n");
    std::cout << std::format("  • Score Head:       16 -> 1 (Sigmoid) -> μ\n");
    std::cout << std::format("  • Uncertainty Head: 16 -> 1 (Softplus) -> σ²\n");
    std::cout << std::format("  • Total de Parâmetros: {}\n", model.num_params());

    auto data_split = tso::DatasetGenerator::generate_exp003(300, kSeed);
    std::cout << std::format("  • Dataset com Missingness: Treino = {}, Val = {}, Test = {}\n\n",
                             data_split.train.size(), data_split.val.size(), data_split.test.size());

    tso::AdamWConfig opt_cfg{
        .lr = 0.005f,
        .beta1 = 0.9f,
        .beta2 = 0.999f,
        .eps = 1e-8f,
        .weight_decay = 0.0005f
    };
    tso::AdamW optimizer(opt_cfg);

    constexpr std::size_t kEpochs = 60;
    constexpr std::size_t kBatchSize = 16;

    std::cout << "\033[1m[2] Treinamento Conjunto com Perda Gaussian NLL (AdamW lr=0.005)\033[0m\n";
    std::cout << "-------------------------------------------------------------------------------------------------------\n";
    std::cout << std::format("{:>6} | {:>10} | {:>10} | {:>10} | {:>10} | {:>9} | {:>10}\n",
                             "Época", "Loss Total", "Choice Loss", "Noul Loss", "NLL Loss", "Choice Acc", "Noul Acc");
    std::cout << "-------------------------------------------------------------------------------------------------------\n";

    for (std::size_t epoch = 1; epoch <= kEpochs; ++epoch) {
        rng.shuffle(data_split.train);

        float epoch_total_loss = 0.0f;
        float epoch_choice_loss = 0.0f;
        float epoch_noul_loss = 0.0f;
        float epoch_nll_loss = 0.0f;

        for (std::size_t i = 0; i < data_split.train.size(); i += kBatchSize) {
            const std::size_t current_batch_size = std::min(kBatchSize, data_split.train.size() - i);
            model.zero_grad();

            for (std::size_t b = 0; b < current_batch_size; ++b) {
                const auto& sample = data_split.train[i + b];
                auto out = model.forward(sample.x);

                auto [c_loss, d_c] = tso::CrossEntropyLoss::compute_from_index(out.choice_probs, sample.y);
                auto [n_loss, d_n] = tso::CrossEntropyLoss::compute_from_index(out.noul_probs, sample.noul_target);
                auto nll = tso::GaussianNLLLoss::compute(out.score, out.uncertainty, sample.score_target);

                const float total_sample_loss = c_loss + n_loss + nll.loss;
                epoch_total_loss += total_sample_loss;
                epoch_choice_loss += c_loss;
                epoch_noul_loss += n_loss;
                epoch_nll_loss += nll.loss;

                const float scale = 1.0f / static_cast<float>(current_batch_size);
                for (auto& g : d_c) g *= scale;
                for (auto& g : d_n) g *= scale;
                for (auto& g : nll.d_mu) g *= scale;
                for (auto& g : nll.d_var) g *= scale;

                model.backward(d_c, d_n, nll.d_mu, nll.d_var);
            }

            model.update(optimizer);
        }

        const float N = static_cast<float>(data_split.train.size());
        const float avg_tot = epoch_total_loss / N;
        const float avg_c = epoch_choice_loss / N;
        const float avg_n = epoch_noul_loss / N;
        const float avg_nll = epoch_nll_loss / N;

        if (epoch % 5 == 0 || epoch == 1 || epoch == kEpochs) {
            std::size_t c_correct = 0;
            std::size_t n_correct = 0;
            for (const auto& sample : data_split.val) {
                auto out = model.forward(sample.x);
                auto max_c = std::max_element(out.choice_probs.begin(), out.choice_probs.end());
                auto max_n = std::max_element(out.noul_probs.begin(), out.noul_probs.end());
                if (static_cast<std::size_t>(std::distance(out.choice_probs.begin(), max_c)) == sample.y) ++c_correct;
                if (static_cast<std::size_t>(std::distance(out.noul_probs.begin(), max_n)) == sample.noul_target) ++n_correct;
            }

            const float val_c_acc = static_cast<float>(c_correct) / static_cast<float>(data_split.val.size()) * 100.0f;
            const float val_n_acc = static_cast<float>(n_correct) / static_cast<float>(data_split.val.size()) * 100.0f;

            std::cout << std::format("{:>6} | {:>10.4f} | {:>11.4f} | {:>9.4f} | {:>10.4f} | {:>9.2f}% | {:>9.2f}%\n",
                                     epoch, avg_tot, avg_c, avg_n, avg_nll, val_c_acc, val_n_acc);
        }
    }
    std::cout << "-------------------------------------------------------------------------------------------------------\n\n";

    // 3. Test Set Evaluation
    std::cout << "\033[1m[3] Avaliação Global no Conjunto de Teste (sob Missingness Misto)\033[0m\n";
    std::vector<tso::Vector> test_c_probs;
    std::vector<tso::Vector> test_n_probs;
    std::vector<std::size_t> test_c_targets;
    std::vector<std::size_t> test_n_targets;
    float total_score_mae = 0.0f;
    float total_unc_mae = 0.0f;

    for (const auto& sample : data_split.test) {
        auto out = model.forward(sample.x);
        test_c_probs.push_back(out.choice_probs);
        test_n_probs.push_back(out.noul_probs);
        test_c_targets.push_back(sample.y);
        test_n_targets.push_back(sample.noul_target);

        total_score_mae += std::abs(out.score - sample.score_target);
        total_unc_mae += std::abs(out.uncertainty - sample.uncertainty_target);
    }

    const float test_N = static_cast<float>(data_split.test.size());
    auto c_report = tso::Calibration::evaluate(test_c_probs, test_c_targets, 10);
    auto n_report = tso::Calibration::evaluate(test_n_probs, test_n_targets, 10);

    std::cout << std::format("  • Choice Head: Acurácia = \033[1;32m{:.2f}%\033[0m | ECE = \033[1;32m{:.4f}\033[0m\n",
                             c_report.accuracy * 100.0f, c_report.ece);
    std::cout << std::format("  • Noul Head:   Acurácia = \033[1;32m{:.2f}%\033[0m | ECE = \033[1;32m{:.4f}\033[0m\n",
                             n_report.accuracy * 100.0f, n_report.ece);
    std::cout << std::format("  • Score Head:  MAE = \033[1;32m{:.4f}\033[0m\n", total_score_mae / test_N);
    std::cout << std::format("  • Incerteza:   MAE = \033[1;32m{:.4f}\033[0m\n\n", total_unc_mae / test_N);

    // 4. Teste de Degradação de Informação: Omissão Progressiva (0% -> 100%)
    std::cout << "\033[1m[4] Experimento de Degradação: Resposta do Modelo sob Omissão Progressiva\033[0m\n";
    std::cout << "Observando se a Incerteza σ² e a Entropia de Shannon H(P) crescem monotonicamente:\n\n";

    // Base Nominal State
    tso::StructuredState nominal_base{
        .declared = true,
        .registered = true,
        .runtime = tso::RuntimeState::Running,
        .witness = tso::WitnessState::Valid,
        .freshness = tso::FreshnessState::Fresh,
        .health = tso::HealthState::Healthy
    };

    std::cout << std::format("{:<15} | {:<12} | {:>10} | {:>12} | {:>9} | {:>12}\n",
                             "Omissão", "Choice Top-1", "Confiança", "Entropia H(P)", "Score μ", "Incerteza σ²");
    std::cout << "------------------------------------------------------------------------------------\n";

    for (std::size_t missing_count = 0; missing_count <= 6; ++missing_count) {
        tso::PresenceMask mask;
        if (missing_count >= 1) mask.health = false;
        if (missing_count >= 2) mask.freshness = false;
        if (missing_count >= 3) mask.witness = false;
        if (missing_count >= 4) mask.runtime = false;
        if (missing_count >= 5) mask.registered = false;
        if (missing_count >= 6) mask.declared = false;

        tso::Vector x = nominal_base.encode_with_mask(mask);
        auto out = model.forward(x);

        auto max_c = std::max_element(out.choice_probs.begin(), out.choice_probs.end());
        std::size_t pred_c = static_cast<std::size_t>(std::distance(out.choice_probs.begin(), max_c));
        const float conf = *max_c;
        const float entropy = tso::Calibration::normalized_entropy(out.choice_probs);

        const float pct = (static_cast<float>(missing_count) / 6.0f) * 100.0f;
        std::string label = std::format("{} campos ({:.0f}%)", missing_count, pct);

        std::cout << std::format("{:<15} | {:<12} | {:>9.1f}% | {:>12.4f} | {:>9.3f} | \033[1;33m{:>12.4f}\033[0m\n",
                                 label, tso::to_string(static_cast<tso::Choice>(pred_c)),
                                 conf * 100.0f, entropy, out.score, out.uncertainty);
    }
    std::cout << "------------------------------------------------------------------------------------\n\n";

    std::cout << "\033[1;32m[✓] Experimento EXP-003 concluído com sucesso!\033[0m\n";
    return 0;
}
