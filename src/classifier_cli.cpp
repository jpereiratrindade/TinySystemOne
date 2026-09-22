#include "tso/classifier.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <string>
#include <string_view>
#include <filesystem>

namespace {

void print_header() {
    std::cout << "======================================================================\n";
    std::cout << "                 TinySystemOne — Classifier CLI                       \n";
    std::cout << "        High-Level Inference, Diagnostics & Benchmark Tool (v2.0.0)   \n";
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

void print_text_result(std::string_view label, std::string_view text, const tso::ClassificationResult& res) {
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << std::format("📝 Enunciado Textual: \033[1;36m\"{}\"\033[0m\n", text);
    std::cout << std::format("🏷️  Contexto: {}\n", label);
    std::cout << "  ──────────────────────────────────────────────────────────────────\n";
    std::cout << std::format("  • Julgamento (Choice):   \033[1;32m{:<12}\033[0m (Confiança: {:.1f}%, Entropia: {:.4f})\n",
                             tso::to_string(res.choice), res.confidence * 100.0f, res.entropy);
    std::cout << std::format("  • Locus Diagnóstico:     \033[1;33m{:<12}\033[0m\n", tso::to_string(res.locus));
    std::cout << std::format("  • Proposição Noul P(true): \033[1;33m{:.4f}\033[0m\n", res.noul_truth_probability);
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
    std::cout << std::format("⚡ Executando Benchmark Vetorial ({} inferências consecutivas)...\n", num_inferences);
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

    std::cout << std::format("  • Duração Total:          {:.2f} ms\n", total_duration_ms);
    std::cout << std::format("  • Throughput de Produção: \033[1;32m{:.0f} inferências/segundo\033[0m\n", throughput);
    std::cout << std::format("  • Latência Média:         \033[1;36m{:.1f} ns ({:.3f} µs)\033[0m\n", avg_ns, avg_ns / 1000.0);
    std::cout << std::format("  • Latência Mínima:        {:.1f} ns ({:.3f} µs)\n", min_ns, min_ns / 1000.0);
    std::cout << std::format("  • Percentil 50 (Mediana): {:.1f} ns ({:.3f} µs)\n", p50_ns, p50_ns / 1000.0);
    std::cout << std::format("  • Percentil 90:           {:.1f} ns ({:.3f} µs)\n", p90_ns, p90_ns / 1000.0);
    std::cout << std::format("  • Percentil 99:           {:.1f} ns ({:.3f} µs)\n", p99_ns, p99_ns / 1000.0);
    std::cout << std::format("  • Latência Máxima:        {:.1f} ns ({:.3f} µs)\n", max_ns, max_ns / 1000.0);
    std::cout << "======================================================================\n";
}

void run_text_benchmark(const tso::SystemOneTextClassifier& text_classifier, std::size_t num_inferences = 50000) {
    std::cout << "\n======================================================================\n";
    std::cout << std::format("⚡ Executando Benchmark Text Transformer ({} inferências)...\n", num_inferences);
    std::cout << "======================================================================\n";

    std::string sample_text = "declaration is true and registration is true. runtime process is running and witness is valid. freshness is fresh and health is healthy.";

    // Warm-up
    for (int i = 0; i < 500; ++i) {
        auto _ = text_classifier.classify(sample_text);
    }

    std::vector<double> latencies;
    latencies.reserve(num_inferences);

    const auto t_total_start = std::chrono::high_resolution_clock::now();

    for (std::size_t i = 0; i < num_inferences; ++i) {
        auto res = text_classifier.classify(sample_text);
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

    std::cout << std::format("  • Duração Total:          {:.2f} ms\n", total_duration_ms);
    std::cout << std::format("  • Throughput Textual:     \033[1;32m{:.0f} inferências/segundo\033[0m\n", throughput);
    std::cout << std::format("  • Latência Média:         \033[1;36m{:.1f} ns ({:.3f} µs)\033[0m\n", avg_ns, avg_ns / 1000.0);
    std::cout << std::format("  • Latência Mínima:        {:.1f} ns ({:.3f} µs)\n", min_ns, min_ns / 1000.0);
    std::cout << std::format("  • Percentil 50 (Mediana): {:.1f} ns ({:.3f} µs)\n", p50_ns, p50_ns / 1000.0);
    std::cout << std::format("  • Percentil 90:           {:.1f} ns ({:.3f} µs)\n", p90_ns, p90_ns / 1000.0);
    std::cout << std::format("  • Percentil 99:           {:.1f} ns ({:.3f} µs)\n", p99_ns, p99_ns / 1000.0);
    std::cout << std::format("  • Latência Máxima:        {:.1f} ns ({:.3f} µs)\n", max_ns, max_ns / 1000.0);
    std::cout << "======================================================================\n";
}

} // namespace

int main(int argc, char* argv[]) {
    print_header();

    // Check for CLI text & question flags: --text "<string>" [--question "<string>"]
    std::string text_input;
    std::string question_input;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "--text" && i + 1 < argc) {
            text_input = argv[++i];
        } else if (arg == "--question" && i + 1 < argc) {
            question_input = argv[++i];
        }
    }

    const std::string model_file = "tso_model.bin";
    const std::string text_model_file = "tso_text_model.bin";

    if (!text_input.empty()) {
        tso::SystemOneTextClassifier text_classifier = [&]() {
            if (std::filesystem::exists(text_model_file)) {
                return tso::SystemOneTextClassifier::load(text_model_file);
            }
            auto split = tso::DatasetGenerator::generate_canonical_split(101);
            auto tc = tso::SystemOneTextClassifier::train_and_calibrate(split);
            tc.save(text_model_file);
            return tc;
        }();

        if (!question_input.empty()) {
            auto res = text_classifier.ask(text_input, question_input);
            print_text_result(std::format("Consulta Tipada (v2.0): \"{}\"", question_input), text_input, res);
        } else {
            auto res = text_classifier.classify(text_input);
            print_text_result("Inferência CLI Direta", text_input, res);
        }
        return 0;
    }

    std::cout << "[1] Treinando e calibrando Classificador Vetorial no Universo Canônico (Ω=576)...\n";
    auto split = tso::DatasetGenerator::generate_canonical_split(101);
    auto classifier = tso::SystemOneClassifier::train_and_calibrate(split);

    std::cout << std::format("  ✓ Classificador vetorial pronto! Parâmetros: {}, T*: {:.4f}, τ_OOD: {:.4f}\n",
                             classifier.identity().parameter_count,
                             classifier.identity().calibrated_temperature,
                             classifier.identity().ood_energy_threshold);

    classifier.save(model_file);
    auto loaded_classifier = tso::SystemOneClassifier::load(model_file);

    std::cout << "\n[2] Treinando e calibrando SystemOneTextClassifier (Pre-LN Transformer)...\n";
    auto text_classifier = tso::SystemOneTextClassifier::train_and_calibrate(split);
    std::cout << std::format("  ✓ Classificador textual pronto! Parâmetros: {}, T*: {:.4f}, τ_OOD: {:.4f}\n",
                             text_classifier.identity().parameter_count,
                             text_classifier.identity().calibrated_temperature,
                             text_classifier.identity().ood_energy_threshold);

    text_classifier.save(text_model_file);
    auto loaded_text_classifier = tso::SystemOneTextClassifier::load(text_model_file);

    std::cout << "\n[3] Executando Julgamentos sobre Observações Estruturadas:\n";

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

    std::cout << "\n[4] Executando Julgamentos sobre Enunciados Textuais Multiestilo:\n";

    // Texto 1: Prosa Natural Nominal
    std::string text_nom = "declaration is true and registration is true. runtime process is running and witness is valid. freshness is fresh and health is healthy.";
    print_text_result("Prosa Natural Nominal", text_nom, loaded_text_classifier.classify(text_nom));

    // Texto 2: Telemetria Log com Falha
    std::string text_fail = "[status] decl=true | reg=true | run=running | wit=valid | fresh=fresh | hlth=failing";
    print_text_result("Log de Telemetria com Falha Crítica", text_fail, loaded_text_classifier.classify(text_fail));

    // Texto 3: Relatório Diagnóstico com Contradição
    std::string text_contra = "service declaration: false; discovery: true; execution: running; verification: valid; telemetry: fresh; metric: healthy";
    print_text_result("Relatório com Contradição de Registro", text_contra, loaded_text_classifier.classify(text_contra));

    // Texto 4: Enunciado OOD Alienígena
    std::string text_alien = "quantum particle flux density oscillation in hyperdrive core anomaly";
    print_text_result("Enunciado Alienígena OOD", text_alien, loaded_text_classifier.classify(text_alien));

    // Run Benchmarks
    run_benchmark(loaded_classifier, 100000);
    run_text_benchmark(loaded_text_classifier, 50000);

    return 0;
}
