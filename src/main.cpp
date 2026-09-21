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

void print_banner() {
    std::cout << "\033[1;36m";
    std::cout << "======================================================================\n";
    std::cout << "                 TinySystemOne — Experiment EXP-001                   \n";
    std::cout << "  Learning Calibrated Probabilistic Judgments in Structured States    \n";
    std::cout << "======================================================================\033[0m\n\n";
}

void print_confusion_matrix(const std::vector<tso::Vector>& probs, const std::vector<std::size_t>& targets) {
    std::size_t matrix[4][4] = {0};
    for (std::size_t i = 0; i < probs.size(); ++i) {
        auto max_it = std::max_element(probs[i].begin(), probs[i].end());
        std::size_t pred = static_cast<std::size_t>(std::distance(probs[i].begin(), max_it));
        matrix[targets[i]][pred]++;
    }

    const char* labels[4] = {"NOMINAL", "DEGRADED", "INCONSISTENT", "UNKNOWN"};
    std::cout << "\n\033[1;33m--- Matriz de Confusão (Linha = Real, Coluna = Predição) ---\033[0m\n";
    std::cout << std::format("{:<14} | {:<10} {:<10} {:<14} {:<10}\n", "Real \\ Pred", labels[0], labels[1], labels[2], labels[3]);
    std::cout << "-----------------------------------------------------------------\n";
    for (int r = 0; r < 4; ++r) {
        std::cout << std::format("{:<14} | {:<10} {:<10} {:<14} {:<10}\n",
                                 labels[r], matrix[r][0], matrix[r][1], matrix[r][2], matrix[r][3]);
    }
    std::cout << "-----------------------------------------------------------------\n";
}

int main() {
    print_banner();

    // 1. Setup PRNG & Hyperparameters
    constexpr std::uint64_t kSeed = 42;
    tso::Random rng(kSeed);

    std::cout << "\033[1m[1] Configuração do Microcosmo & Arquitetura\033[0m\n";
    std::vector<tso::LayerConfig> topology = {
        {18, 32, tso::Activation::GELU},
        {32, 16, tso::Activation::GELU},
        {16, 4,  tso::Activation::None}
    };

    tso::MLP model(topology, rng);
    std::cout << std::format("  • Topologia: 18 (Input) -> 32 (GELU) -> 16 (GELU) -> 4 (Softmax)\n");
    std::cout << std::format("  • Total de Parâmetros: {}\n", model.num_params());

    // 2. Generate Dataset
    auto data_split = tso::DatasetGenerator::generate_exp001(250, kSeed);
    std::cout << std::format("  • Dataset: Treino = {}, Validação = {}, Teste = {}, OOD = {}\n\n",
                             data_split.train.size(), data_split.val.size(), data_split.test.size(), data_split.ood.size());

    // 3. Optimizer Setup
    tso::AdamWConfig opt_cfg{
        .lr = 0.008f,
        .beta1 = 0.9f,
        .beta2 = 0.999f,
        .eps = 1e-8f,
        .weight_decay = 0.001f
    };
    tso::AdamW optimizer(opt_cfg);

    constexpr std::size_t kEpochs = 60;
    constexpr std::size_t kBatchSize = 16;

    std::cout << "\033[1m[2] Treinamento (AdamW lr=0.008, weight_decay=0.001)\033[0m\n";
    std::cout << "--------------------------------------------------------------------------------------\n";
    std::cout << std::format("{:>6} | {:>10} | {:>10} | {:>10} | {:>10} | {:>12}\n",
                             "Época", "Loss Treino", "Acc Treino", "Acc Val", "ECE Val", "Entropia Val");
    std::cout << "--------------------------------------------------------------------------------------\n";

    for (std::size_t epoch = 1; epoch <= kEpochs; ++epoch) {
        rng.shuffle(data_split.train);

        float total_loss = 0.0f;
        std::size_t train_correct = 0;

        // Mini-batch training
        for (std::size_t i = 0; i < data_split.train.size(); i += kBatchSize) {
            const std::size_t current_batch_size = std::min(kBatchSize, data_split.train.size() - i);
            model.zero_grad();

            for (std::size_t b = 0; b < current_batch_size; ++b) {
                const auto& sample = data_split.train[i + b];
                tso::Vector probs = model.forward(sample.x);
                auto [sample_loss, dlogits] = tso::CrossEntropyLoss::compute_from_index(probs, sample.y);
                
                total_loss += sample_loss;
                auto max_it = std::max_element(probs.begin(), probs.end());
                if (static_cast<std::size_t>(std::distance(probs.begin(), max_it)) == sample.y) {
                    ++train_correct;
                }

                // Accumulate gradients scaled by 1/batch_size
                for (auto& g : dlogits) g /= static_cast<float>(current_batch_size);
                model.backward(dlogits);
            }

            model.update(optimizer);
        }

        const float avg_train_loss = total_loss / static_cast<float>(data_split.train.size());
        const float train_acc = static_cast<float>(train_correct) / static_cast<float>(data_split.train.size());

        // Periodic evaluation on validation set
        if (epoch % 5 == 0 || epoch == 1 || epoch == kEpochs) {
            std::vector<tso::Vector> val_probs;
            std::vector<std::size_t> val_targets;
            val_probs.reserve(data_split.val.size());
            val_targets.reserve(data_split.val.size());

            for (const auto& sample : data_split.val) {
                val_probs.push_back(model.forward(sample.x));
                val_targets.push_back(sample.y);
            }

            auto val_report = tso::Calibration::evaluate(val_probs, val_targets, 10);
            std::cout << std::format("{:>6} | {:>10.4f} | {:>9.2f}% | {:>9.2f}% | {:>10.4f} | {:>12.4f}\n",
                                     epoch, avg_train_loss, train_acc * 100.0f, val_report.accuracy * 100.0f,
                                     val_report.ece, val_report.avg_entropy);
        }
    }
    std::cout << "--------------------------------------------------------------------------------------\n\n";

    // 4. Test Set Evaluation & Calibration Analysis
    std::cout << "\033[1m[3] Avaliação no Conjunto de Teste (Dados Inéditos no Treino)\033[0m\n";
    std::vector<tso::Vector> test_probs;
    std::vector<std::size_t> test_targets;
    test_probs.reserve(data_split.test.size());
    test_targets.reserve(data_split.test.size());

    for (const auto& sample : data_split.test) {
        test_probs.push_back(model.forward(sample.x));
        test_targets.push_back(sample.y);
    }

    auto test_report = tso::Calibration::evaluate(test_probs, test_targets, 10);

    std::cout << std::format("  • Acurácia Global de Julgamento: \033[1;32m{:.2f}%\033[0m\n", test_report.accuracy * 100.0f);
    std::cout << std::format("  • Expected Calibration Error (ECE): \033[1;32m{:.4f}\033[0m\n", test_report.ece);
    std::cout << std::format("  • Mean Brier Score: {:.4f}\n", test_report.brier_score);
    std::cout << std::format("  • Entropia Média de Shannon: {:.4f} nats\n", test_report.avg_entropy);
    std::cout << std::format("  • Confiança Média: {:.2f}%\n", test_report.avg_confidence * 100.0f);

    print_confusion_matrix(test_probs, test_targets);

    // Calibration Bins Table
    std::cout << "\n\033[1;33m--- Tabela de Bins de Calibração (Diagrama de Confiabilidade) ---\033[0m\n";
    std::cout << std::format("{:<12} | {:>8} | {:>12} | {:>12} | {:>10}\n",
                             "Faixa Conf", "Qtd", "Conf Média", "Acurácia", "Gap ECE");
    std::cout << "-----------------------------------------------------------------\n";
    for (const auto& bin : test_report.bins) {
        if (bin.count > 0) {
            std::cout << std::format("[{:.1f}, {:.1f}]   | {:>8} | {:>11.2f}% | {:>11.2f}% | {:>10.4f}\n",
                                     bin.conf_lower, bin.conf_upper, bin.count,
                                     bin.avg_confidence * 100.0f, bin.accuracy * 100.0f, bin.calibration_gap);
        }
    }
    std::cout << "-----------------------------------------------------------------\n\n";

    // 5. Out-of-Distribution (OOD) & Extreme Ambiguity Test
    std::cout << "\033[1m[4] Teste de Julgamento sob Incerteza e OOD (Out-Of-Distribution)\033[0m\n";
    std::cout << "Investigando se o modelo emite entropia e incerteza honesta sob evidência caótica ou ausente:\n\n";

    for (std::size_t i = 0; i < std::min(std::size_t(6), data_split.ood.size()); ++i) {
        const auto& ood_sample = data_split.ood[i];
        tso::Vector probs = model.forward(ood_sample.x);
        
        auto max_it = std::max_element(probs.begin(), probs.end());
        std::size_t pred_idx = static_cast<std::size_t>(std::distance(probs.begin(), max_it));
        float conf = *max_it;
        float norm_entropy = tso::Calibration::normalized_entropy(probs);

        std::cout << std::format("\033[1mCenário OOD #{}:\033[0m {}\n", i + 1, ood_sample.description);
        std::cout << std::format("  ↳ Distribuição: [NOMINAL: {:.3f}, DEGRADED: {:.3f}, INCONSISTENT: {:.3f}, UNKNOWN: {:.3f}]\n",
                                 probs[0], probs[1], probs[2], probs[3]);
        std::cout << std::format("  ↳ Escolha Top-1: \033[1;35m{}\033[0m (Confiança: {:.1f}%, Entropia Normalizada: {:.2f})\n\n",
                                 tso::to_string(static_cast<tso::Choice>(pred_idx)), conf * 100.0f, norm_entropy);
    }

    std::cout << "\033[1;32m[✓] Experimento EXP-001 concluído com sucesso!\033[0m\n";
    return 0;
}
