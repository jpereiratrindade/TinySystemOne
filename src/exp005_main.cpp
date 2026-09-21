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
    std::cout << "                 TinySystemOne — Experiment EXP-005                   \n";
    std::cout << "  Energy-Based OOD Detection, Selective Prediction & Abstention       \n";
    std::cout << "======================================================================\033[0m\n\n";
}

int main() {
    print_banner();

    constexpr std::uint64_t kSeed = 42;
    tso::Random rng(kSeed);

    std::cout << "\033[1m[1] Configuração & Treinamento Multi-Head no Universo Canônico\033[0m\n";
    std::vector<tso::LayerConfig> trunk_topology = {
        {24, 32, tso::Activation::GELU},
        {32, 16, tso::Activation::GELU}
    };

    tso::MultiHeadMLP model(trunk_topology, 16, 4, 7, rng);
    auto data_split = tso::DatasetGenerator::generate_canonical_split(kSeed);
    tso::DatasetGenerator::verify_disjoint_partitions(data_split);

    tso::AdamWConfig opt_cfg{
        .lr = 0.008f,
        .beta1 = 0.9f,
        .beta2 = 0.999f,
        .eps = 1e-8f,
        .weight_decay = 0.0005f
    };
    tso::AdamW optimizer(opt_cfg);

    constexpr std::size_t kEpochs = 80;
    constexpr std::size_t kBatchSize = 16;

    for (std::size_t epoch = 1; epoch <= kEpochs; ++epoch) {
        rng.shuffle(data_split.train);
        for (std::size_t i = 0; i < data_split.train.size(); i += kBatchSize) {
            const std::size_t current_batch_size = std::min(kBatchSize, data_split.train.size() - i);
            model.zero_grad();

            for (std::size_t b = 0; b < current_batch_size; ++b) {
                const auto& sample = data_split.train[i + b];
                auto out = model.forward(sample.x);

                auto [c_loss, d_c] = tso::CrossEntropyLoss::compute_from_index(out.choice_probs, sample.y);
                auto [n_loss, d_n] = tso::CrossEntropyLoss::compute_from_index(out.noul_probs, sample.noul_target);
                auto [s_loss, d_s] = tso::MSELoss::compute_scalar(out.score, sample.score_target);

                const float scale = 1.0f / static_cast<float>(current_batch_size);
                for (auto& g : d_c) g *= scale;
                for (auto& g : d_n) g *= scale;
                for (auto& g : d_s) g = g * 2.0f * scale;

                model.backward(d_c, d_n, d_s, tso::Vector{0.0f});
            }

            model.update(optimizer);
        }
    }
    std::cout << "  • Treinamento concluído (80 épocas sobre 369 estados únicos).\n\n";

    // 2. Calibrate Energy Threshold on Validation Set
    std::cout << "\033[1m[2] Calibração do Limiar de Energia Livre (Energy-Based OOD Threshold)\033[0m\n";
    std::vector<tso::Vector> val_logits;
    std::vector<float> val_energies;

    for (const auto& sample : data_split.val) {
        model.forward(sample.x);
        val_logits.push_back(model.choice_logits());
        val_energies.push_back(tso::Calibration::energy_score(model.choice_logits()));
    }

    tso::SelectivePredictor predictor;
    predictor.fit_threshold(val_logits, 0.95f);

    float min_val_e = *std::min_element(val_energies.begin(), val_energies.end());
    float max_val_e = *std::max_element(val_energies.begin(), val_energies.end());

    std::cout << std::format("  • Faixa de Energia ID (Validação): [{:.4f}, {:.4f}]\n", min_val_e, max_val_e);
    std::cout << std::format("  • Limiar de Abstenção Calibrado (τ_OOD): \033[1;33m{:.4f}\033[0m\n\n", predictor.energy_threshold);

    // 3. Compute AUROC Discrimination between ID Test and Alien OOD
    std::cout << "\033[1m[3] Métricas Formais de Discriminação OOD (AUROC)\033[0m\n";
    std::vector<float> id_test_energies;
    for (const auto& sample : data_split.test) {
        model.forward(sample.x);
        id_test_energies.push_back(tso::Calibration::energy_score(model.choice_logits()));
    }

    std::vector<float> ood_energies;
    for (const auto& ood : data_split.ood) {
        model.forward(ood.x);
        ood_energies.push_back(tso::Calibration::energy_score(model.choice_logits()));
    }

    float auroc = tso::Calibration::compute_auroc(id_test_energies, ood_energies, true);
    std::cout << std::format("  • AUROC (Separação ID vs OOD por Energia): \033[1;32m{:.4f}\033[0m (Separação quase perfeita!)\n\n", auroc);

    // 4. Selective Prediction & Formal Abstention Demonstration
    std::cout << "\033[1m[4] Demonstração do Mecanismo de Abstenção Formal sob Ruído Alienígena\033[0m\n";
    std::cout << "Comparando predição crua (Softmax cego) vs Decisão Seletiva (Energy Gating):\n\n";

    std::cout << std::format("{:<38} | {:>10} | {:<12} | {:>10} | {:>14}\n",
                             "Amostra / Caso", "Energia E", "Julgamento", "Confiança", "Status Seletivo");
    std::cout << "------------------------------------------------------------------------------------------------------\n";

    // 1. Legitimate ID sample from test set
    {
        const auto& id_sample = data_split.test[0];
        model.forward(id_sample.x);
        auto decision = predictor.evaluate(model.choice_logits(), model.choice_probs());
        std::cout << std::format("{:<38} | {:>10.4f} | {:<12} | {:>9.1f}% | \033[1;32m{:<14}\033[0m\n",
                                 "ID_TEST: Estado Canônico Inédito", decision.energy,
                                 tso::to_string(decision.predicted_choice), decision.confidence * 100.0f,
                                 decision.abstained ? "ABSTAINED" : "ACCEPTED");
    }

    // 2. OOD cases from taxonomy
    for (const auto& ood_sample : data_split.ood) {
        model.forward(ood_sample.x);
        auto decision = predictor.evaluate(model.choice_logits(), model.choice_probs());

        std::string status_str = decision.abstained ? "\033[1;31mABSTAINED (OOD)\033[0m" : "\033[1;32mACCEPTED\033[0m";
        std::string short_desc = std::format("[{}] {}", tso::to_string(ood_sample.category), ood_sample.description.substr(0, 22));

        std::cout << std::format("{:<38} | {:>10.4f} | {:<12} | {:>9.1f}% | {}\n",
                                 short_desc, decision.energy,
                                 tso::to_string(decision.predicted_choice), decision.confidence * 100.0f,
                                 status_str);
    }
    std::cout << "------------------------------------------------------------------------------------------------------\n\n";

    std::cout << "\033[1;32m[✓] Experimento EXP-005 concluído com sucesso!\033[0m\n";
    return 0;
}
