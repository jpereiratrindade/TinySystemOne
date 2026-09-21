#include "tso/classifier.hpp"
#include "tso/dataset.hpp"
#include "tso/calibration.hpp"
#include <iostream>
#include <iomanip>
#include <format>
#include <vector>
#include <numeric>
#include <algorithm>
#include <string>
#include <filesystem>

using namespace tso;

int main() {
    std::cout << "======================================================================\n";
    std::cout << "                 TinySystemOne — Experiment EXP-010                   \n";
    std::cout << "  Milestone v1.0.0: Machine-Native Typed Judgment over Textual States \n";
    std::cout << "======================================================================\n\n";

    constexpr std::uint64_t kSeed = 101;
    auto split = DatasetGenerator::generate_canonical_split(kSeed);
    auto canonical_universe = DatasetGenerator::generate_canonical_universe();

    std::cout << "\033[1m[1] Espaço Canônico & Corpus Multiestilo de Enunciados Textuais\033[0m\n";
    std::cout << std::format("  • Universo Canônico Base:     {} configurações de estado (Ω=576)\n", canonical_universe.size());
    std::cout << std::format("  • Formatos de Sintaxe:        4 estilos (Prosa, Telemetria, Diagnóstico, Compacto)\n");
    std::cout << std::format("  • Corpus Textual Total:       {} enunciados semi-estruturados\n", canonical_universe.size() * 4);
    std::cout << std::format("  • Divisão Disjunta:           Treino: {} | Validação: {} | Teste: {}\n\n",
                             split.train.size() * 4, split.val.size() * 4, split.test.size() * 4);

    // 2. End-to-End Training and Calibration
    std::cout << "\033[1m[2] Treinamento End-to-End e Calibração Pós-Hoc do SystemOneTextClassifier\033[0m\n";
    const auto t_train_start = std::chrono::high_resolution_clock::now();

    TextClassifierTrainingConfig cfg{
        .epochs = 90,
        .batch_size = 16,
        .learning_rate = 0.006f,
        .weight_decay = 0.0005f,
        .ood_percentile = 0.98f,
        .seed = kSeed
    };

    auto classifier = SystemOneTextClassifier::train_and_calibrate(split, cfg);

    const auto t_train_end = std::chrono::high_resolution_clock::now();
    const double train_duration_ms = std::chrono::duration<double, std::milli>(t_train_end - t_train_start).count();

    std::cout << std::format("  ✓ Treinamento concluído em {:.2f} ms ({:.2f} s)\n", train_duration_ms, train_duration_ms / 1000.0);
    std::cout << std::format("  • Parâmetros Totais:          {}\n", classifier.identity().parameter_count);
    std::cout << std::format("  • Temperatura Calibrada (T*): {:.4f}\n", classifier.identity().calibrated_temperature);
    std::cout << std::format("  • Limiar de Energia OOD (τ):  {:.4f}\n\n", classifier.identity().ood_energy_threshold);

    // 3. Serialization to Disk & Clean Deserialization
    const std::string export_model_path = "tso_v1_text_production.bin";
    std::cout << "\033[1m[3] Serialização Binária e Recarga Independente de Runtime\033[0m\n";
    classifier.save(export_model_path);
    std::cout << std::format("  ✓ Modelo serializado para '{}' ({} bytes)\n",
                             export_model_path, std::filesystem::file_size(export_model_path));

    auto runtime_classifier = SystemOneTextClassifier::load(export_model_path);
    std::cout << "  ✓ Runtime carregado de arquivo binário sem qualquer dependência externa!\n\n";

    // 4. Out-of-Sample Performance across All 4 Styles on Held-out Test Set
    std::cout << "\033[1m[4] Avaliação Out-of-Sample nos 4 Estilos Textuais (Conjunto de Teste)\033[0m\n";

    std::unordered_map<std::size_t, StructuredState> state_map;
    for (const auto& s : canonical_universe) {
        state_map[s.state_id()] = s;
    }

    const std::vector<std::pair<TextualStyle, std::string>> styles = {
        {TextualStyle::NaturalProse, "Prosa Natural"},
        {TextualStyle::TelemetryLog, "Log de Telemetria"},
        {TextualStyle::DiagnosticReport, "Relatório Diagnóstico"},
        {TextualStyle::CompactKeyValue, "Pares Chave-Valor"}
    };

    for (const auto& [style, name] : styles) {
        std::size_t correct_choice = 0;
        std::size_t correct_locus = 0;
        std::vector<Vector> all_choice_probs;
        std::vector<std::size_t> all_choice_targets;

        for (const auto& sample : split.test) {
            const auto& st = state_map[sample.state_id];
            std::string text = TextualStateGenerator::generate_text(st, style);
            auto res = runtime_classifier.classify(text);

            bool c_ok = (static_cast<std::size_t>(res.choice) == sample.y);
            bool l_ok = (static_cast<std::size_t>(res.locus) == sample.noul_target);

            if (c_ok) ++correct_choice;
            if (l_ok) ++correct_locus;

            Vector probs(res.choice_probabilities.begin(), res.choice_probabilities.end());
            all_choice_probs.push_back(probs);
            all_choice_targets.push_back(sample.y);
        }

        const double choice_acc = static_cast<double>(correct_choice) / static_cast<double>(split.test.size()) * 100.0;
        const double locus_acc = static_cast<double>(correct_locus) / static_cast<double>(split.test.size()) * 100.0;
        const auto rep = Calibration::evaluate(all_choice_probs, all_choice_targets, 10);

        std::cout << std::format("  • {:<24} | Choice Acc: \033[1;32m{:6.2f}%\033[0m | Locus Acc: {:6.2f}% | ECE: {:.4f} | H(P): {:.4f}\n",
                                 name, choice_acc, locus_acc, rep.ece, rep.avg_entropy);
    }
    std::cout << "\n";

    // 5. Contradiction and Degradation Case Studies
    std::cout << "\033[1m[5] Estudos de Caso de Julgamento Tipado em Tempo Real\033[0m\n";

    std::vector<std::pair<std::string, std::string>> case_studies = {
        {"1. Nominal Saudável", "declaration is true and registration is true. runtime process is running and witness is valid. freshness is fresh and health is healthy."},
        {"2. Contradição de Registro", "service declaration: false; discovery: true; execution: running; verification: valid; telemetry: fresh; metric: healthy"},
        {"3. Degradação de Freshness", "[status] decl=true | reg=true | run=running | wit=valid | fresh=expired | hlth=healthy"},
        {"4. Falha Crítica de Processo", "decl:true reg:true run:absent wit:valid fresh:fresh hlth:failing"},
        {"5. Pergunta/Texto Alienígena OOD", "quantum phase variance oscillation detected in subsystem 7"}
    };

    for (const auto& [title, text] : case_studies) {
        auto res = runtime_classifier.classify(text);
        std::cout << std::format("  🔷 \033[1m{}\033[0m\n", title);
        std::cout << std::format("     Texto: \"{}\"\n", text);
        std::cout << std::format("     -> Choice: \033[1;32m{:<12}\033[0m (Conf: {:.1f}%, H: {:.4f}) | Locus: {:<12} | Score: {:.3f} | E(x): {:.4f} | Status: {}\n\n",
                                 to_string(res.choice), res.confidence * 100.0f, res.entropy,
                                 to_string(res.locus), res.score, res.free_energy, res.decision_status);
    }

    // 6. High-Throughput Latency Benchmark
    std::cout << "\033[1m[6] Benchmark de Latência e Throughput em Produção\033[0m\n";
    constexpr std::size_t kInferences = 50000;
    std::string bench_sample = "declaration is true and registration is true. runtime process is running and witness is valid. freshness is fresh and health is healthy.";

    std::vector<double> latencies;
    latencies.reserve(kInferences);

    // Warm-up
    for (int i = 0; i < 500; ++i) {
        auto _ = runtime_classifier.classify(bench_sample);
    }

    const auto t_bench_start = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < kInferences; ++i) {
        auto res = runtime_classifier.classify(bench_sample);
        latencies.push_back(res.latency_nanoseconds);
    }
    const auto t_bench_end = std::chrono::high_resolution_clock::now();
    const double bench_ms = std::chrono::duration<double, std::milli>(t_bench_end - t_bench_start).count();

    std::sort(latencies.begin(), latencies.end());
    const double sum_ns = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    const double avg_ns = sum_ns / static_cast<double>(kInferences);
    const double p50_ns = latencies[kInferences * 50 / 100];
    const double p90_ns = latencies[kInferences * 90 / 100];
    const double p99_ns = latencies[kInferences * 99 / 100];
    const double throughput = static_cast<double>(kInferences) / (bench_ms / 1000.0);

    std::cout << std::format("  • Throughput Textual:     \033[1;32m{:.0f} avaliações/segundo\033[0m\n", throughput);
    std::cout << std::format("  • Latência Média:         \033[1;36m{:.1f} ns ({:.3f} µs)\033[0m\n", avg_ns, avg_ns / 1000.0);
    std::cout << std::format("  • Latência Mediana (p50): {:.1f} ns ({:.3f} µs)\n", p50_ns, p50_ns / 1000.0);
    std::cout << std::format("  • Percentil 90 (p90):     {:.1f} ns ({:.3f} µs)\n", p90_ns, p90_ns / 1000.0);
    std::cout << std::format("  • Percentil 99 (p99):     {:.1f} ns ({:.3f} µs)\n\n", p99_ns, p99_ns / 1000.0);

    std::cout << "======================================================================\n";
    std::cout << "  ✓ MILESTONE v1.0.0 ALCANÇADO: TinySystemOne Operacional em Produção!\n";
    std::cout << "======================================================================\n";

    return 0;
}
