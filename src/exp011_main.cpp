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
    std::cout << "                 TinySystemOne — Experiment EXP-011                   \n";
    std::cout << " Milestone v2.0.0: Question-Conditioned Typed Judgment & Proposition Noul\n";
    std::cout << "======================================================================\n\n";

    constexpr std::uint64_t kSeed = 101;
    auto split = DatasetGenerator::generate_canonical_split(kSeed);
    auto canonical_universe = DatasetGenerator::generate_canonical_universe();

    std::cout << "\033[1m[1] Espaço Canônico & Corpus de Perguntas Tipadas Dinâmicas\033[0m\n";
    std::cout << std::format("  • Universo Canônico Base:     {} estados (Ω=576)\n", canonical_universe.size());
    std::cout << "  • Primitivas Tipadas (v2.0):  Choice, Locus, Continuous Score e Proposition Noul P(prop=true)\n";
    std::cout << "  • Modos de Consulta:          Dual-Segment [CLS] State [SEP] Question [SEP] [PAD]\n\n";

    // 2. Training Question-Conditioned Transformer
    std::cout << "\033[1m[2] Treinamento End-to-End do Transformer com Autoatenção Cruzada QA\033[0m\n";
    const auto t_train_start = std::chrono::high_resolution_clock::now();

    TextClassifierTrainingConfig cfg{
        .epochs = 80,
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

    // 3. Dynamic Question Answering Demonstration on Identical States
    std::cout << "\033[1m[3] Demonstração: Múltiplas Perguntas Tipadas sobre o Mesmo Estado Textual\033[0m\n";

    // Cenário A: Estado Nominal
    std::string state_nom = "declaration is true and registration is true. runtime process is running and witness is valid. freshness is fresh and health is healthy.";
    std::cout << "──────────────────────────────────────────────────────────────────────\n";
    std::cout << std::format("📍 \033[1mEstado A (Nominal):\033[0m \"{}\"\n", state_nom);
    std::cout << "──────────────────────────────────────────────────────────────────────\n";

    std::vector<std::string> questions_a = {
        "question: what is the overall system state?",
        "question: is witness valid?",
        "question: is process running?",
        "question: rate system vitality and health",
        "question: which component caused the failure?"
    };

    for (const auto& q : questions_a) {
        auto res = classifier.ask(state_nom, q);
        std::cout << std::format("  ❓ \033[1;36m{}\033[0m\n", q);
        std::cout << std::format("     -> Choice: \033[1;32m{:<12}\033[0m (Conf: {:.1f}%) | Noul P(true): \033[1;33m{:.3f}\033[0m | Locus: {:<10} | Score: {:.3f} ({:.1f} µs)\n",
                                 to_string(res.choice), res.confidence * 100.0f,
                                 res.noul_truth_probability, to_string(res.locus), res.score, res.latency_nanoseconds / 1000.0);
    }
    std::cout << "\n";

    // Cenário B: Estado com Contradição de Declaração
    std::string state_contra = "service declaration: false; discovery: true; execution: running; verification: valid; telemetry: fresh; metric: healthy";
    std::cout << "──────────────────────────────────────────────────────────────────────\n";
    std::cout << std::format("📍 \033[1mEstado B (Contradição):\033[0m \"{}\"\n", state_contra);
    std::cout << "──────────────────────────────────────────────────────────────────────\n";

    std::vector<std::string> questions_b = {
        "question: what is the overall system state?",
        "question: is declaration confirmed?",
        "question: which component caused the failure?",
        "question: is witness valid?"
    };

    for (const auto& q : questions_b) {
        auto res = classifier.ask(state_contra, q);
        std::cout << std::format("  ❓ \033[1;36m{}\033[0m\n", q);
        std::cout << std::format("     -> Choice: \033[1;32m{:<12}\033[0m (Conf: {:.1f}%) | Noul P(true): \033[1;33m{:.3f}\033[0m | Locus: \033[1;31m{:<10}\033[0m | Score: {:.3f} ({:.1f} µs)\n",
                                 to_string(res.choice), res.confidence * 100.0f,
                                 res.noul_truth_probability, to_string(res.locus), res.score, res.latency_nanoseconds / 1000.0);
    }
    std::cout << "\n";

    // 4. Evaluation over Held-out Test Set QA pairs
    std::cout << "\033[1m[4] Avaliação de Acurácia nos Pares (Estado, Pergunta) do Conjunto de Teste\033[0m\n";

    std::unordered_map<std::size_t, StructuredState> state_map;
    for (const auto& s : canonical_universe) {
        state_map[s.state_id()] = s;
    }

    std::size_t total_qa = 0;
    std::size_t correct_choice_qa = 0;
    std::size_t total_props = 0;
    double prop_mae = 0.0;

    for (const auto& sample : split.test) {
        const auto& st = state_map[sample.state_id];
        std::string state_text = TextualStateGenerator::generate_text(st, TextualStyle::NaturalProse);
        auto questions = QuestionGenerator::generate_questions(st);

        for (const auto& q : questions) {
            auto res = classifier.ask(state_text, q.text);
            ++total_qa;

            if (!q.is_proposition) {
                if (res.choice == q.expected_choice) ++correct_choice_qa;
            } else {
                ++total_props;
                prop_mae += std::abs(res.noul_truth_probability - q.expected_noul_prob);
            }
        }
    }

    const double choice_qa_acc = static_cast<double>(correct_choice_qa) / static_cast<double>(total_qa - total_props) * 100.0;
    const double avg_prop_mae = prop_mae / static_cast<double>(total_props);

    std::cout << std::format("  • Acurácia em Perguntas de Escolha e Locus: \033[1;32m{:6.2f}%\033[0m (N={})\n",
                             choice_qa_acc, total_qa - total_props);
    std::cout << std::format("  • Erro Médio Absoluto (MAE) em Proposições Noul P(true): \033[1;32m{:.4f}\033[0m (N={})\n\n",
                             avg_prop_mae, total_props);

    // 5. High-Throughput Latency Benchmark
    std::cout << "\033[1m[5] Benchmark de Latência e Throughput de Question-Answering\033[0m\n";
    constexpr std::size_t kInferences = 50000;
    std::string bench_q = "question: is witness valid?";

    std::vector<double> latencies;
    latencies.reserve(kInferences);

    for (int i = 0; i < 500; ++i) {
        auto _ = classifier.ask(state_nom, bench_q);
    }

    const auto t_bench_start = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < kInferences; ++i) {
        auto res = classifier.ask(state_nom, bench_q);
        latencies.push_back(res.latency_nanoseconds);
    }
    const auto t_bench_end = std::chrono::high_resolution_clock::now();
    const double bench_ms = std::chrono::duration<double, std::milli>(t_bench_end - t_bench_start).count();

    std::sort(latencies.begin(), latencies.end());
    const double sum_ns = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    const double avg_ns = sum_ns / static_cast<double>(kInferences);
    const double p50_ns = latencies[kInferences * 50 / 100];
    const double p99_ns = latencies[kInferences * 99 / 100];
    const double throughput = static_cast<double>(kInferences) / (bench_ms / 1000.0);

    std::cout << std::format("  • Throughput QA:          \033[1;32m{:.0f} consultas/segundo\033[0m\n", throughput);
    std::cout << std::format("  • Latência Média:         \033[1;36m{:.1f} ns ({:.3f} µs)\033[0m\n", avg_ns, avg_ns / 1000.0);
    std::cout << std::format("  • Latência Mediana (p50): {:.1f} ns ({:.3f} µs)\n", p50_ns, p50_ns / 1000.0);
    std::cout << std::format("  • Percentil 99 (p99):     {:.1f} ns ({:.3f} µs)\n\n", p99_ns, p99_ns / 1000.0);

    std::cout << "======================================================================\n";
    std::cout << "  ✓ MILESTONE v2.0.0 ALCANÇADO: TinySystemOne Roadmap 100% Concluído!\n";
    std::cout << "======================================================================\n";

    return 0;
}
