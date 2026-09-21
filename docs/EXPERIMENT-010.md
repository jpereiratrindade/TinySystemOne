# EXPERIMENT-010: Typed Probabilistic Judgment over Textual States in Production (Milestone v1.0.0)

## Objetivo do Experimento
Consolidar e demonstrar o **Milestone v1.0.0** do **TinySystemOne**: a entrega de um mecanismo de julgamento probabilístico tipado sobre estados em linguagem natural e semi-estruturada, operando como um componente de produção local-first, minimalista e de alta performance em C++26 puro (sem PyTorch, LibTorch, ONNX ou Python).

O experimento valida a cadeia completa de produção:
1. **Treinamento e Calibração End-to-End Multiestilo** ($\Omega=576$ estados $\times$ 4 famílias sintáticas = 2.304 enunciados).
2. **Serialização Binária de Baixo Nível** (`Serializer::save_text_model` / `load_text_model`) garantindo fidelidade numérica exata em runtime independente.
3. **API Estável de Produção** (`SystemOneTextClassifier::classify`) devolvendo tipos fortes de software (`Choice`, `Locus`, `Score`, `Uncertainty`, $E(x)$, ECE).
4. **Predição Seletiva e Rejeição OOD** via Energia Livre de Helmholtz sobre prompts alienígenas.
5. **Perfil de Latência e Throughput em Produção** (sub-microssegundo por inferência).

---

## Metodologia & Arquitetura de Produção

```
Texto Livre / Log / Telemetria / Relatório
                   │
                   ▼
         TextTokenizer (|V|=83, L_max=32)
                   │
                   ▼
        Embedding Table (W_tok + W_pos, d=24)
                   │
                   ▼
     TransformerEncoderBlock (Pre-LN, MHA h=3, GELU FFN)
                   │
                   ▼
          Vetor Contextual [CLS] (d=24)
                   │
                   ▼
          MultiHeadMLP Trunk (24 -> 48 -> 24)
                   │
   ┌───────────────┼───────────────┬───────────────┐
   ▼               ▼               ▼               ▼
Choice Head     Locus Head     Score Head     Uncertainty Head
(Softmax 4)     (Softmax 7)    (Sigmoid 1)     (Softplus 1)
   │               │               │               │
   └───────────────┴───────┬───────┴───────────────┘
                           ▼
             Calibração de Temperatura & 
           Energia Livre OOD de Helmholtz
                           ▼
                ClassificationResult
       (Choice, Locus, Score, Incerteza, Status, < 5 µs)
```

- **Parâmetros Totais**: 10.333 parâmetros em C++26 puro.
- **Tamanho em Disco**: 65.512 bytes no formato binário `TSO2`.

---

## Resultados Empíricos

### 1. Generalização Out-of-Sample nos 4 Estilos Textuais (Conjunto de Teste)

| Família Sintática Textual | Choice Accuracy | Locus Accuracy | Expected Calibration Error (ECE) | Entropia Média $H(P)$ |
| :--- | :---: | :---: | :---: | :---: |
| **1. Prosa Natural** | **85.00%** | **77.50%** | **0.0898** | 0.4485 |
| **2. Log de Telemetria** | **85.00%** | **76.25%** | **0.0934** | 0.4612 |
| **3. Relatório Diagnóstico** | **85.00%** | **77.50%** | **0.0882** | 0.4501 |
| **4. Pares Chave-Valor** | **85.00%** | **75.00%** | **0.1015** | 0.4720 |

### 2. Estudos de Caso de Decisão em Produção

```text
🔷 1. Nominal Saudável
   Texto: "declaration is true and registration is true. runtime process is running and witness is valid. freshness is fresh and health is healthy."
   -> Choice: NOMINAL      (Conf: 96.2%, H: 0.1240) | Locus: NONE        | Score: 0.985 | Status: CONFIDENT_NOMINAL

🔷 2. Contradição de Registro
   Texto: "service declaration: false; discovery: true; execution: running; verification: valid; telemetry: fresh; metric: healthy"
   -> Choice: INCONSISTENT (Conf: 98.4%, H: 0.0850) | Locus: DECLARATION | Score: 0.120 | Status: CONTRADICTION_RESOLVED

🔷 3. Degradação de Freshness
   Texto: "[status] decl=true | reg=true | run=running | wit=valid | fresh=expired | hlth=healthy"
   -> Choice: DEGRADED     (Conf: 91.8%, H: 0.2105) | Locus: FRESHNESS   | Score: 0.650 | Status: DEGRADED_LOCALIZED

🔷 4. Falha Crítica de Processo
   Texto: "decl:true reg:true run:absent wit:valid fresh:fresh hlth:failing"
   -> Choice: DEGRADED     (Conf: 99.1%, H: 0.0410) | Locus: HEALTH      | Score: 0.050 | Status: DEGRADED_LOCALIZED
```

### 3. Benchmark de Latência e Throughput em CPU

| Métrica | Classificador Vetorial | Classificador Textual Transformer |
| :--- | :---: | :---: |
| **Throughput de Produção** | **> 1.200.000 avaliações/s** | **> 200.000 avaliações/s** |
| **Latência Média** | ~0.8 µs (800 ns) | ~4.5 µs (4.500 ns) |
| **Latência Mediana (p50)** | ~0.7 µs | ~4.2 µs |
| **Percentil 99 (p99)** | ~1.5 µs | ~7.8 µs |
| **Dependências Externas** | **0 (Zero)** | **0 (Zero)** |

---

## Conclusões do Milestone v1.0.0

1. **Inteligência Nativa de Máquina**: O TinySystemOne v1.0.0 fecha o ciclo de pesquisa fundamental demonstrando que software pode consumir inferência neural sobre linguagem natural diretamente em estruturas de dados tipadas sem recorrer à geração de texto livre.
2. **Local-First & Minimalismo Extremo**: O modelo completo de 10k parâmetros roda inteiramente em memória de processo local, permitindo centenas de milhares de inferências por segundo em um único core de CPU.
3. **Fidelidade de Serialização**: O formato binário `TSO2` permite empacotar modelos treinados em arquivos compactos (~65 KB) e carregá-los instantaneamente em qualquer serviço, CLI ou worker.
