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
    std::cout << "                 TinySystemOne — Experiment EXP-004                   \n";
    std::cout << "  Contradictory Evidence, Conflict Resolution & Temperature Scaling   \n";
    std::cout << "======================================================================\033[0m\n\n";
}

int main() {
    print_banner();

    constexpr std::uint64_t kSeed = 42;
    tso::Random rng(kSeed);

    std::cout << "\033[1m[1] Configuração & Treinamento Multi-Head\033[0m\n";
    std::vector<tso::LayerConfig> trunk_topology = {
        {24, 32, tso::Activation::GELU},
        {32, 16, tso::Activation::GELU}
    };

    tso::MultiHeadMLP model(trunk_topology, 16, 4, 7, rng);
    auto data_split = tso::DatasetGenerator::generate_exp001(300, kSeed);

    tso::AdamWConfig opt_cfg{
        .lr = 0.006f,
        .beta1 = 0.9f,
        .beta2 = 0.999f,
        .eps = 1e-8f,
        .weight_decay = 0.0005f
    };
    tso::AdamW optimizer(opt_cfg);

    constexpr std::size_t kEpochs = 60;
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
    std::cout << "  • Treinamento concluído (60 épocas).\n\n";

    // 2. Fit Post-Hoc Temperature Scaling on Validation Set
    std::cout << "\033[1m[2] Ajuste do Calibrador Post-Hoc (Temperature Scaling no Conjunto de Validação)\033[0m\n";
    std::vector<tso::Vector> val_choice_logits;
    std::vector<tso::Vector> val_noul_logits;
    std::vector<std::size_t> val_choice_targets;
    std::vector<std::size_t> val_noul_targets;

    for (const auto& sample : data_split.val) {
        model.forward(sample.x);
        val_choice_logits.push_back(model.choice_logits());
        val_noul_logits.push_back(model.noul_logits());
        val_choice_targets.push_back(sample.y);
        val_noul_targets.push_back(sample.noul_target);
    }

    tso::TemperatureScaler choice_scaler;
    choice_scaler.fit(val_choice_logits, val_choice_targets, 150, 0.05f);

    tso::TemperatureScaler noul_scaler;
    noul_scaler.fit(val_noul_logits, val_noul_targets, 150, 0.05f);

    std::cout << std::format("  • Temperatura Ótima Choice (T*): \033[1;33m{:.4f}\033[0m\n", choice_scaler.temperature);
    std::cout << std::format("  • Temperatura Ótima Noul   (T*): \033[1;33m{:.4f}\033[0m\n\n", noul_scaler.temperature);

    // 3. Test Set Comparison: Before vs After Calibration
    std::cout << "\033[1m[3] Comparação no Conjunto de Teste: Pré vs Pós-Calibração por Temperatura\033[0m\n";
    std::vector<tso::Vector> raw_choice_probs;
    std::vector<tso::Vector> calibrated_choice_probs;
    std::vector<std::size_t> test_targets;

    for (const auto& sample : data_split.test) {
        model.forward(sample.x);
        raw_choice_probs.push_back(model.choice_probs());
        calibrated_choice_probs.push_back(choice_scaler.calibrate(model.choice_logits()));
        test_targets.push_back(sample.y);
    }

    auto report_raw = tso::Calibration::evaluate(raw_choice_probs, test_targets, 10);
    auto report_cal = tso::Calibration::evaluate(calibrated_choice_probs, test_targets, 10);

    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << std::format("{:<24} | {:>15} | {:>15}\n", "Métrica de Calibração", "Logits Brutos", "Pós-Temperature Scaling");
    std::cout << "------------------------------------------------------------------------------------\n";
    std::cout << std::format("{:<24} | {:>14.2f}% | {:>14.2f}%\n", "Acurácia Global", report_raw.accuracy * 100.0f, report_cal.accuracy * 100.0f);
    std::cout << std::format("{:<24} | {:>15.4f} | \033[1;32m{:>15.4f}\033[0m\n", "ECE (Calibration Error)", report_raw.ece, report_cal.ece);
    std::cout << std::format("{:<24} | {:>15.4f} | \033[1;32m{:>15.4f}\033[0m\n", "Brier Score", report_raw.brier_score, report_cal.brier_score);
    std::cout << std::format("{:<24} | {:>15.4f} | {:>15.4f}\n", "Entropia Média (nats)", report_raw.avg_entropy, report_cal.avg_entropy);
    std::cout << std::format("{:<24} | {:>14.2f}% | {:>14.2f}%\n", "Confiança Média", report_raw.avg_confidence * 100.0f, report_cal.avg_confidence * 100.0f);
    std::cout << "------------------------------------------------------------------------------------\n\n";

    // 4. Contradictory Evidence & Conflict Resolution Battery
    std::cout << "\033[1m[4] Bateria de Resolução de Conflitos e Evidência Contraditória\033[0m\n";
    std::cout << "Testando a capacidade de identificar contradições e atribuir o locus causal correto:\n\n";

    struct ConflictCase {
        std::string title;
        tso::StructuredState state;
        std::string expected_nature;
    };

    std::vector<ConflictCase> cases = {
        {
            "Conflito de Declaração / Registro",
            tso::StructuredState{.declared = false, .registered = true, .runtime = tso::RuntimeState::Running, .witness = tso::WitnessState::Valid, .freshness = tso::FreshnessState::Fresh, .health = tso::HealthState::Healthy},
            "Entidade registrada sem declaração de existência"
        },
        {
            "Conflito Runtime vs Quórum de Testemunha",
            tso::StructuredState{.declared = true, .registered = true, .runtime = tso::RuntimeState::Running, .witness = tso::WitnessState::Invalid, .freshness = tso::FreshnessState::Fresh, .health = tso::HealthState::Healthy},
            "Runtime ativo porém assinatura/testemunho revogado ou inválido"
        },
        {
            "Conflito de Integridade Temporal",
            tso::StructuredState{.declared = true, .registered = true, .runtime = tso::RuntimeState::Running, .witness = tso::WitnessState::Valid, .freshness = tso::FreshnessState::Expired, .health = tso::HealthState::Healthy},
            "Saúde reportada como saudável porém dados completamente expirados"
        },
        {
            "Discordância de Presença (Ausente mas Testemunha Válida)",
            tso::StructuredState{.declared = true, .registered = true, .runtime = tso::RuntimeState::Absent, .witness = tso::WitnessState::Valid, .freshness = tso::FreshnessState::Fresh, .health = tso::HealthState::Healthy},
            "Processo ausente mas atestado como válido pelo auditor"
        }
    };

    for (std::size_t i = 0; i < cases.size(); ++i) {
        const auto& c = cases[i];
        tso::Vector x = c.state.encode();
        auto out = model.forward(x);

        tso::Vector cal_c_probs = choice_scaler.calibrate(model.choice_logits());
        tso::Vector cal_n_probs = noul_scaler.calibrate(model.noul_logits());

        auto max_c = std::max_element(cal_c_probs.begin(), cal_c_probs.end());
        auto max_n = std::max_element(cal_n_probs.begin(), cal_n_probs.end());
        std::size_t pred_c = static_cast<std::size_t>(std::distance(cal_c_probs.begin(), max_c));
        std::size_t pred_n = static_cast<std::size_t>(std::distance(cal_n_probs.begin(), max_n));

        std::cout << std::format("\033[1mCenário Contraditório #{}:\033[0m {}\n", i + 1, c.title);
        std::cout << std::format("  ↳ Natureza: {}\n", c.expected_nature);
        std::cout << std::format("  ↳ Julgamento: \033[1;35m{}\033[0m (Conf: {:.1f}%) | Locus Causal (Noul): \033[1;33m{}\033[0m (Conf: {:.1f}%) | Score: \033[1;32m{:.3f}\033[0m\n\n",
                                 tso::to_string(static_cast<tso::Choice>(pred_c)), *max_c * 100.0f,
                                 tso::to_string(static_cast<tso::Noul>(pred_n)), *max_n * 100.0f,
                                 out.score);
    }

    std::cout << "\033[1;32m[✓] Experimento EXP-004 concluído com sucesso!\033[0m\n";
    return 0;
}
