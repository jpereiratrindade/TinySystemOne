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
    std::cout << "                 TinySystemOne — Experiment EXP-002                   \n";
    std::cout << "  Multi-Head Triple Judgment: Choice + Noul (Causal) + Continuous Score\n";
    std::cout << "======================================================================\033[0m\n\n";
}

void print_choice_confusion(const std::vector<tso::Vector>& probs, const std::vector<std::size_t>& targets) {
    std::size_t matrix[4][4] = {0};
    for (std::size_t i = 0; i < probs.size(); ++i) {
        auto max_it = std::max_element(probs[i].begin(), probs[i].end());
        std::size_t pred = static_cast<std::size_t>(std::distance(probs[i].begin(), max_it));
        matrix[targets[i]][pred]++;
    }

    const char* labels[4] = {"NOMINAL", "DEGRADED", "INCONSISTENT", "UNKNOWN"};
    std::cout << "\n\033[1;33m--- Matriz de Confusão: Choice ---\033[0m\n";
    std::cout << std::format("{:<14} | {:<10} {:<10} {:<14} {:<10}\n", "Real \\ Pred", labels[0], labels[1], labels[2], labels[3]);
    std::cout << "-----------------------------------------------------------------\n";
    for (int r = 0; r < 4; ++r) {
        std::cout << std::format("{:<14} | {:<10} {:<10} {:<14} {:<10}\n",
                                 labels[r], matrix[r][0], matrix[r][1], matrix[r][2], matrix[r][3]);
    }
    std::cout << "-----------------------------------------------------------------\n";
}

void print_noul_confusion(const std::vector<tso::Vector>& probs, const std::vector<std::size_t>& targets) {
    std::size_t matrix[7][7] = {0};
    for (std::size_t i = 0; i < probs.size(); ++i) {
        auto max_it = std::max_element(probs[i].begin(), probs[i].end());
        std::size_t pred = static_cast<std::size_t>(std::distance(probs[i].begin(), max_it));
        matrix[targets[i]][pred]++;
    }

    const char* labels[7] = {"NONE", "DECL", "RUN", "WIT", "FRESH", "HLTH", "MULT"};
    std::cout << "\n\033[1;33m--- Matriz de Confusão: Noul (Locus Causal) ---\033[0m\n";
    std::cout << std::format("{:<8} | {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6}\n",
                             "Real\\Pred", labels[0], labels[1], labels[2], labels[3], labels[4], labels[5], labels[6]);
    std::cout << "-----------------------------------------------------------------\n";
    for (int r = 0; r < 7; ++r) {
        std::cout << std::format("{:<8} | {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6}\n",
                                 labels[r], matrix[r][0], matrix[r][1], matrix[r][2], matrix[r][3], matrix[r][4], matrix[r][5], matrix[r][6]);
    }
    std::cout << "-----------------------------------------------------------------\n";
}

int main() {
    print_banner();

    constexpr std::uint64_t kSeed = 42;
    tso::Random rng(kSeed);

    std::cout << "\033[1m[1] Configuração da Arquitetura Multi-Head\033[0m\n";
    std::vector<tso::LayerConfig> trunk_topology = {
        {24, 32, tso::Activation::GELU},
        {32, 16, tso::Activation::GELU}
    };

    tso::MultiHeadMLP model(trunk_topology, 16, 4, 7, rng);
    std::cout << std::format("  • Trunk: 24 -> 32 (GELU) -> 16 (GELU)\n");
    std::cout << std::format("  • Choice Head: 16 -> 4 (Softmax)\n");
    std::cout << std::format("  • Noul Head:   16 -> 7 (Softmax)\n");
    std::cout << std::format("  • Score Head:  16 -> 1 (Sigmoid)\n");
    std::cout << std::format("  • Total de Parâmetros: {}\n", model.num_params());

    auto data_split = tso::DatasetGenerator::generate_exp001(250, kSeed);
    std::cout << std::format("  • Dataset: Treino = {}, Validação = {}, Teste = {}, OOD = {}\n\n",
                             data_split.train.size(), data_split.val.size(), data_split.test.size(), data_split.ood.size());

    tso::AdamWConfig opt_cfg{
        .lr = 0.006f,
        .beta1 = 0.9f,
        .beta2 = 0.999f,
        .eps = 1e-8f,
        .weight_decay = 0.0005f
    };
    tso::AdamW optimizer(opt_cfg);

    constexpr std::size_t kEpochs = 70;
    constexpr std::size_t kBatchSize = 16;

    std::cout << "\033[1m[2] Treinamento Conjunto Multi-Head (AdamW lr=0.006)\033[0m\n";
    std::cout << "-----------------------------------------------------------------------------------------------\n";
    std::cout << std::format("{:>6} | {:>10} | {:>10} | {:>10} | {:>9} | {:>9} | {:>10}\n",
                             "Época", "Loss Total", "Choice Loss", "Noul Loss", "Score MSE", "Choice Acc", "Noul Acc");
    std::cout << "-----------------------------------------------------------------------------------------------\n";

    for (std::size_t epoch = 1; epoch <= kEpochs; ++epoch) {
        rng.shuffle(data_split.train);

        float epoch_total_loss = 0.0f;
        float epoch_choice_loss = 0.0f;
        float epoch_noul_loss = 0.0f;
        float epoch_score_mse = 0.0f;

        for (std::size_t i = 0; i < data_split.train.size(); i += kBatchSize) {
            const std::size_t current_batch_size = std::min(kBatchSize, data_split.train.size() - i);
            model.zero_grad();

            for (std::size_t b = 0; b < current_batch_size; ++b) {
                const auto& sample = data_split.train[i + b];
                auto out = model.forward(sample.x);

                auto [c_loss, d_c] = tso::CrossEntropyLoss::compute_from_index(out.choice_probs, sample.y);
                auto [n_loss, d_n] = tso::CrossEntropyLoss::compute_from_index(out.noul_probs, sample.noul_target);
                auto [s_loss, d_s] = tso::MSELoss::compute_scalar(out.score, sample.score_target);

                const float total_sample_loss = c_loss + n_loss + 2.0f * s_loss;
                epoch_total_loss += total_sample_loss;
                epoch_choice_loss += c_loss;
                epoch_noul_loss += n_loss;
                epoch_score_mse += s_loss;

                const float scale = 1.0f / static_cast<float>(current_batch_size);
                for (auto& g : d_c) g *= scale;
                for (auto& g : d_n) g *= scale;
                for (auto& g : d_s) g = g * 2.0f * scale;

                model.backward(d_c, d_n, d_s, tso::Vector{0.0f});
            }

            model.update(optimizer);
        }

        const float N = static_cast<float>(data_split.train.size());
        const float avg_tot = epoch_total_loss / N;
        const float avg_c = epoch_choice_loss / N;
        const float avg_n = epoch_noul_loss / N;
        const float avg_s = epoch_score_mse / N;

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

            std::cout << std::format("{:>6} | {:>10.4f} | {:>11.4f} | {:>9.4f} | {:>9.4f} | {:>9.2f}% | {:>9.2f}%\n",
                                     epoch, avg_tot, avg_c, avg_n, avg_s, val_c_acc, val_n_acc);
        }
    }
    std::cout << "-----------------------------------------------------------------------------------------------\n\n";

    // 3. Test Set Evaluation
    std::cout << "\033[1m[3] Avaliação Tripla no Conjunto de Teste\033[0m\n";
    std::vector<tso::Vector> test_c_probs;
    std::vector<tso::Vector> test_n_probs;
    std::vector<std::size_t> test_c_targets;
    std::vector<std::size_t> test_n_targets;
    float total_score_mse = 0.0f;
    float total_score_mae = 0.0f;

    for (const auto& sample : data_split.test) {
        auto out = model.forward(sample.x);
        test_c_probs.push_back(out.choice_probs);
        test_n_probs.push_back(out.noul_probs);
        test_c_targets.push_back(sample.y);
        test_n_targets.push_back(sample.noul_target);

        const float diff = std::abs(out.score - sample.score_target);
        total_score_mse += diff * diff;
        total_score_mae += diff;
    }

    const float test_N = static_cast<float>(data_split.test.size());
    auto c_report = tso::Calibration::evaluate(test_c_probs, test_c_targets, 10);
    auto n_report = tso::Calibration::evaluate(test_n_probs, test_n_targets, 10);

    std::cout << std::format("  • Choice Head: Acurácia = \033[1;32m{:.2f}%\033[0m | ECE = \033[1;32m{:.4f}\033[0m | Brier = {:.4f}\n",
                             c_report.accuracy * 100.0f, c_report.ece, c_report.brier_score);
    std::cout << std::format("  • Noul Head:   Acurácia = \033[1;32m{:.2f}%\033[0m | ECE = \033[1;32m{:.4f}\033[0m | Brier = {:.4f}\n",
                             n_report.accuracy * 100.0f, n_report.ece, n_report.brier_score);
    std::cout << std::format("  • Score Head:  MSE = \033[1;32m{:.4f}\033[0m | MAE = \033[1;32m{:.4f}\033[0m\n",
                             total_score_mse / test_N, total_score_mae / test_N);

    print_choice_confusion(test_c_probs, test_c_targets);
    print_noul_confusion(test_n_probs, test_n_targets);

    // 4. Test on OOD Cases
    std::cout << "\n\033[1m[4] Julgamento Triplo sob Incerteza e OOD (Out-Of-Distribution)\033[0m\n";
    for (std::size_t i = 0; i < std::min(std::size_t(5), data_split.ood.size()); ++i) {
        const auto& ood_sample = data_split.ood[i];
        auto out = model.forward(ood_sample.x);

        auto max_c = std::max_element(out.choice_probs.begin(), out.choice_probs.end());
        auto max_n = std::max_element(out.noul_probs.begin(), out.noul_probs.end());
        std::size_t pred_c = static_cast<std::size_t>(std::distance(out.choice_probs.begin(), max_c));
        std::size_t pred_n = static_cast<std::size_t>(std::distance(out.noul_probs.begin(), max_n));

        std::cout << std::format("\033[1mOOD #{}:\033[0m {}\n", i + 1, ood_sample.description);
        std::cout << std::format("  ↳ Choice: \033[1;35m{}\033[0m (Conf: {:.1f}%, Entropia: {:.2f})\n",
                                 tso::to_string(static_cast<tso::Choice>(pred_c)), *max_c * 100.0f,
                                 tso::Calibration::normalized_entropy(out.choice_probs));
        std::cout << std::format("  ↳ Noul:   \033[1;33m{}\033[0m (Conf: {:.1f}%, Entropia: {:.2f})\n",
                                 tso::to_string(static_cast<tso::Noul>(pred_n)), *max_n * 100.0f,
                                 tso::Calibration::normalized_entropy(out.noul_probs));
        std::cout << std::format("  ↳ Score:  \033[1;32m{:.3f}\033[0m (Degradação contínua da saúde operacional)\n\n",
                                 out.score);
    }

    std::cout << "\033[1;32m[✓] Experimento EXP-002 concluído com sucesso!\033[0m\n";
    return 0;
}
