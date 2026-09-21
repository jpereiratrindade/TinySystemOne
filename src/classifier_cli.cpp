#include "tso/classifier.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>

namespace {

void print_header() {
    std::cout << "======================================================================\n";
    std::cout << "                 TinySystemOne — Classifier CLI                       \n";
    std::cout << "        High-Level Inference, Diagnostics & Benchmark Tool            \n";
    std::cout << "======================================================================\n\n";
}

void print_result(std::string_view label, const tso::Observation& obs, const tso::ClassificationResult& res) {
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << std::format("🔷 Cenário: \033[1;36m{}\033[0m\n", label);
    std::cout << std::format("  • Estado: declared={} | reg={} | run={} | wit={} | fresh={} | health={}\n",
                             obs.state.declared, obs.state.registered,
                             tso::to_string(obs.state.runtime), tso::to_string(obs.state.witness),
                             tso::to_string(obs.state.freshness), tso::to_string(obs.state.health));
    std::cout << std::format("  • Sensores Presentes: declared={} | reg={} | run={} | wit={} | fresh={} | health={}\n",
                             obs.mask.declared, obs.mask.registered, obs.mask.runtime,
                             obs.mask.witness, obs.mask.freshness, obs.mask.health);
    std::cout << "  ──────────────────────────────────────────────────────────────────\n";
    std::cout << std::format("  • Julgamento (Choice):   \033[1;32m{:<12}\033[0m (Confiança: {:.1f}%, Entropia: {:.4f})\n",
                             tso::to_string(res.choice), res.confidence * 100.0f, res.entropy);
    std::cout << std::format("  • Locus Diagnóstico:     \033[1;33m{:<12}\033[0m\n", tso::to_string(res.locus));
    std::cout << std::format("  • Viabilidade Contínua:  {:.4f}\n", res.score);
    std::cout << std::format("  • Incerteza Epistêmica:  {:.4f}\n", res.uncertainty);
    std::cout << std::format("  • Energia Livre E(x):    {:.4f} (Limiar OOD: {:.4f})\n", res.free_energy, res.model.ood_energy_threshold);
    std::cout << std::format("  • Status de Decisão:     \033[1;35m{}\033[0m\n", res.decision_status);
    std::cout << std::format("  • Tempo de Inferência:   \033[1;37m{:.1f} ns\033[0m ({:.3f} µs)\n", res.latency_nanoseconds, res.latency_nanoseconds / 1000.0);
    std::cout << "  • Distribuição Choice:   [NOM: "
              << std::format("{:.1f}%, DEG: {:.1f}%, INC: {:.1f}%, UNK: {:.1f}%]\n",
                             res.choice_probabilities[0] * 100.0f,
                             res.choice_probabilities[1] * 100.0f,
                             res.choice_probabilities[2] * 100.0f,
                             res.choice_probabilities[3] * 100.0f);
}

void run_benchmark(const tso::SystemOneClassifier& classifier, std::size_t num_inferences = 100000) {
    std::cout << "\n======================================================================\n";
    std::cout << std::format("⚡ Executando Benchmark de Latência ({} inferências consecutivas)...\n", num_inferences);
    std::cout << "======================================================================\n";

    tso::Observation sample_obs{
        .state = {
            .declared = true,
            .registered = true,
            .runtime = tso::RuntimeState::Running,
            .witness = tso::WitnessState::Valid,
            .freshness = tso::FreshnessState::Fresh,
            .health = tso::HealthState::Healthy
        }
    };

    // Warm-up
    for (int i = 0; i < 1000; ++i) {
        auto _ = classifier.classify(sample_obs);
    }

    std::vector<double> latencies;
    latencies.reserve(num_inferences);

    const auto t_total_start = std::chrono::high_resolution_clock::now();

    for (std::size_t i = 0; i < num_inferences; ++i) {
        auto res = classifier.classify(sample_obs);
        latencies.push_back(res.latency_nanoseconds);
    }

    const auto t_total_end = std::chrono::high_resolution_clock::now();
    const double total_duration_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();

    std::sort(latencies.begin(), latencies.end());

    const double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    const double avg_ns = sum / static_cast<double>(num_inferences);
    const double min_ns = latencies.front();
    const double max_ns = latencies.back();
    const double p50_ns = latencies[num_inferences * 50 / 100];
    const double p90_ns = latencies[num_inferences * 90 / 100];
    const double p99_ns = latencies[num_inferences * 99 / 100];
    const double throughput = static_cast<double>(num_inferences) / (total_duration_ms / 1000.0);

    std::cout << std::format("  • Duração Total:         {:.2f} ms\n", total_duration_ms);
    std::cout << std::format("  • Throughput de Produção: \033[1;32m{:.0f} inferências/segundo\033[0m\n", throughput);
    std::cout << std::format("  • Latência Média:         \033[1;36m{:.1f} ns ({:.3f} µs)\033[0m\n", avg_ns, avg_ns / 1000.0);
    std::cout << std::format("  • Latência Mínima:        {:.1f} ns ({:.3f} µs)\n", min_ns, min_ns / 1000.0);
    std::cout << std::format("  • Percentil 50 (Mediana): {:.1f} ns ({:.3f} µs)\n", p50_ns, p50_ns / 1000.0);
    std::cout << std::format("  • Percentil 90:           {:.1f} ns ({:.3f} µs)\n", p90_ns, p90_ns / 1000.0);
    std::cout << std::format("  • Percentil 99:           {:.1f} ns ({:.3f} µs)\n", p99_ns, p99_ns / 1000.0);
    std::cout << std::format("  • Latência Máxima:        {:.1f} ns ({:.3f} µs)\n", max_ns, max_ns / 1000.0);
    std::cout << "======================================================================\n";
}

} // namespace

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    print_header();

    const std::string model_file = "tso_model.bin";
    std::cout << "[1] Treinando e calibrando SystemOneClassifier no Universo Canônico (Ω=576)...\n";
    auto split = tso::DatasetGenerator::generate_canonical_split(101);
    auto classifier = tso::SystemOneClassifier::train_and_calibrate(split);

    std::cout << std::format("  ✓ Classificador instanciado! Parâmetros: {}, T*: {:.4f}, τ_OOD: {:.4f}\n\n",
                             classifier.identity().parameter_count,
                             classifier.identity().calibrated_temperature,
                             classifier.identity().ood_energy_threshold);

    // Save model to binary file
    std::cout << std::format("[2] Serializando pesos para '{}'...\n", model_file);
    classifier.save(model_file);
    std::cout << "  ✓ Modelo serializado com sucesso!\n\n";

    // Reload from binary file to guarantee consumption integrity
    std::cout << std::format("[3] Recarregando classificador a partir de '{}'...\n", model_file);
    auto loaded_classifier = tso::SystemOneClassifier::load(model_file);
    std::cout << "  ✓ Modelo recarregado com integridade verificada!\n\n";

    std::cout << "[4] Executando Julgamentos sobre Observações Típicas e Anômalas:\n";

    // Caso 1: Nominal Perfeito
    tso::Observation obs_nominal{
        .state = {
            .declared = true,
            .registered = true,
            .runtime = tso::RuntimeState::Running,
            .witness = tso::WitnessState::Valid,
            .freshness = tso::FreshnessState::Fresh,
            .health = tso::HealthState::Healthy
        }
    };
    print_result("Estado Nominal Perfeito", obs_nominal, loaded_classifier.classify(obs_nominal));

    // Caso 2: Degradação Localizada (Witness Stale)
    tso::Observation obs_degraded{
        .state = {
            .declared = true,
            .registered = true,
            .runtime = tso::RuntimeState::Running,
            .witness = tso::WitnessState::Stale,
            .freshness = tso::FreshnessState::Fresh,
            .health = tso::HealthState::Healthy
        }
    };
    print_result("Degradação Localizada (Witness Stale)", obs_degraded, loaded_classifier.classify(obs_degraded));

    // Caso 3: Contradição Factual Direta (Declared=False mas Registered=True e Running)
    tso::Observation obs_contradiction{
        .state = {
            .declared = false,
            .registered = true,
            .runtime = tso::RuntimeState::Running,
            .witness = tso::WitnessState::Valid,
            .freshness = tso::FreshnessState::Fresh,
            .health = tso::HealthState::Healthy
        }
    };
    print_result("Contradição Factual Direta", obs_contradiction, loaded_classifier.classify(obs_contradiction));

    // Caso 4: Omissão de Sensores (Missing Values Explícitos)
    tso::Observation obs_missing{
        .state = {
            .declared = true,
            .registered = true,
            .runtime = tso::RuntimeState::Running,
            .witness = tso::WitnessState::Valid,
            .freshness = tso::FreshnessState::Fresh,
            .health = tso::HealthState::Healthy
        },
        .mask = {
            .declared = true,
            .registered = false, // Sensor Offline
            .runtime = false,    // Sensor Offline
            .witness = false,    // Sensor Offline
            .freshness = false,  // Sensor Offline
            .health = false      // Sensor Offline
        }
    };
    print_result("Perda Crítica de Sensores (5 offline)", obs_missing, loaded_classifier.classify(obs_missing));

    // Caso 5: Ruído Alienígena / OOD Total (Ausência de Sinal)
    tso::Vector null_input(24, 0.0f);
    auto ood_res = loaded_classifier.classify_raw(null_input);
    tso::Observation ood_obs{};
    print_result("Ruído Alienígena (Zero Signal - OOD)", ood_obs, ood_res);

    // Run Microsecond Latency Benchmark
    run_benchmark(loaded_classifier, 100000);

    return 0;
}
