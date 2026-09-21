# API de Classificação do TinySystemOne (`SystemOneClassifier`)

A classe `tso::SystemOneClassifier` é a primeira **superfície de consumo de alto nível** do TinySystemOne. Ela encapsula toda a inteligência neural, o modelo multi-head (1.549 parâmetros), a calibração pós-treinamento ($T^*$) e a porta de energia de Helmholtz para abstenção formal ($\tau_{\text{OOD}}$).

---

## 1. Separação Arquitetural

```text
┌─────────────────────────────────────────────────────────────┐
│                 TinySystemOne Engine                        │
│   (MLP, Forward, Backward, AdamW, NLL, Calibração, OOD)     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 tso::SystemOneClassifier                    │
│   API de Alto Nível de Consumo (Observation → Judgment)     │
└──────────────────────────────┬──────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
┌───────────────────────┐             ┌───────────────────────┐
│     CLI & Benchmark   │             │   Bancada de Triagem  │
│  (tso_classifier)     │             │      (Em Breve)       │
└───────────────────────┘             └───────────────────────┘
```

---

## 2. Contrato de Entrada e Saída

### Entrada: `Observation`
Recebe o estado estruturado do sistema e a máscara de presença dos sensores (`PresenceMask`):

```cpp
#include <tso/classifier.hpp>

tso::Observation obs{
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
        .registered = true,
        .runtime = true,
        .witness = true,
        .freshness = true,
        .health = true
    }
};
```

### Saída: `ClassificationResult`
Retorna um julgamento rico, calibrado e auditável:

```cpp
struct ClassificationResult {
    // 1. Julgamento Tipado Principal
    Choice choice;                       // NOMINAL, DEGRADED, INCONSISTENT, UNKNOWN
    Noul locus;                          // Locus Diagnóstico (NONE, DECLARATION, RUNTIME, ...)

    // 2. Scores Contínuos & Incerteza Epistêmica
    float score;                         // Viabilidade estimada \hat{\mu} \in [0, 1]
    float uncertainty;                   // Variância epistêmica heteroscedástica \hat{\sigma}^2

    // 3. Distribuições de Probabilidade Completas e Calibradas
    std::array<float, 4> choice_probabilities; // Distribuição calibrada com T*
    std::array<float, 7> locus_probabilities;  // Distribuição de atribuição diagnóstica

    // 4. Diagnóstico de Confiança & Detecção OOD
    float confidence;                    // Max P(choice)
    float entropy;                       // Entropia normalizada H(P) \in [0, 1]
    float free_energy;                   // Energia Livre de Helmholtz E(x; T)

    // 5. Roteamento & Decisão Seletiva
    bool abstained;                      // True se for ruído fora do domínio (OOD)
    std::string decision_status;         // Sumário legível (ex: "CONFIDENT_NOMINAL")

    // 6. Proveniência & Performance
    ModelIdentity model;                 // Versão, timestamp, T*, tau_OOD
    double latency_nanoseconds;          // Tempo de inferência em nanossegundos
};
```

---

## 3. Serialização e Persistência Binária (`Serializer`)

O TinySystemOne possui um formato binário proprietário de alto desempenho (`TSO_MODEL_V1`) com cabeçalho mágico `0x54534F31` (`TSO1`):

```cpp
// Salvar modelo treinado e calibrado
classifier.save("tso_model.bin");

// Recarregar em outro processo com fidelidade idêntica bit-a-bit
auto loaded_classifier = tso::SystemOneClassifier::load("tso_model.bin");
```

---

## 4. Benchmark de Latência e Desempenho

Resultados medidos em **100.000 inferências consecutivas** em C++26 puro (Thread única, CPU moderna):

| Métrica | Valor Medido |
|---|---|
| **Throughput de Produção** | **~400.000 inferências / segundo** |
| **Latência Mínima** | `2.00 µs` (`2.004 ns`) |
| **Latência Mediana (P50)** | `2.04 µs` (`2.040 ns`) |
| **Latência Média** | `2.34 µs` (`2.339 ns`) |
| **Percentil 99 (P99)** | `4.16 µs` (`4.163 ns`) |
| **Uso de Memória do Modelo** | `< 10 KB` (1.549 parâmetros Float32) |
