# TinySystemOne

> **Investigando experimentalmente julgamento probabilístico tipado, incerteza e calibração em C++26.**

O **TinySystemOne** é uma implementação white-box de uma arquitetura neural deliberada, minimalista e autônoma, projetada do zero para estudar a emergência de julgamento, confiança e incerteza sobre evidências completas, incompletas e contraditórias.

---

## Características

- 🚀 **100% C++26 puro**: Zero frameworks pesados (sem PyTorch, LibTorch, ONNX), zero dependência de rede, zero dependência de serviços externos.
- 📐 **White-Box Neural Engine**: Álgebra linear, ativadores (GELU, ReLU, Sigmoid, Softplus, Softmax), retropropagação analítica exata e otimizadores (AdamW, SGD) implementados explicitamente.
- 🎯 **Incerteza e Calibração de 1ª Classe**: Shannon Entropy, Brier Score, Expected Calibration Error (ECE) e **Temperature Scaling Post-Hoc** com otimizador analítico em C++26 puro.
- 🧠 **Resolução de Conflitos, Incerteza e OOD (v0.5.0)**:
  - **Choice**: qualidade do estado (`NOMINAL`, `DEGRADED`, `INCONSISTENT`, `UNKNOWN`).
  - **Locus** (diagnóstico): atribuição do locus de conflito/falha (`NONE`, `DECLARATION`, `RUNTIME`, etc.).
  - **Score**: grau contínuo de viabilidade operacional ($\hat{\mu}$).
  - **Uncertainty**: variância heteroscedástica aprendida ($\hat{\sigma}^2$).
  - **Energy-Based OOD & Selective Prediction**: Detecção de ruído alienígena via Helmholtz Free Energy $E(x; T)$ e abstenção formal sem re-treinamento.

---

## Princípios Fundamentais & Arquitetura

O TinySystemOne segue a tese de **Machine-Native Intelligence**: sistemas neurais deliberados para produzir julgamentos tipados e estruturados para consumo direto por software, sem geração de texto livre.

```text
       ┌─────────────────────────────────────────────────────────────┐
       │                   TinySystemOne Engine                      │
       │  Runtime neural local, white-box, C++26, sub-microssegundo  │
       └──────────────────────────────┬──────────────────────────────┘
                                      │
                                      ▼
       ┌─────────────────────────────────────────────────────────────┐
       │                    SystemOneClassifier                      │
       │   API C++ estável: classify(), save(), load(), OOD Filter   │
       └──────────────────────────────┬──────────────────────────────┘
                                      │
                 ┌────────────────────┼────────────────────┐
                 ▼                    ▼                    ▼
          [C++ In-Process]     [CLI / Benchmark]    [Transport Adapters]
          (zero overhead)      (tso_classifier)     (Unix socket / etc)
```

### Diretrizes Transversais

1. **Local-First & Transport-Agnostic**: O TSO é uma biblioteca in-process offline por construção. Adaptadores (CLI, sockets, etc.) são projeções que consomem o contrato C++ sem alterar a semântica do julgamento.
2. **Minimal Sufficient Model**: O crescimento de parâmetros ocorre estritamente para responder a limites experimentais observados (ex: Mean-Pooling $\to$ Self-Attention), mantendo total transparência matemática.

---

## Como Compilar e Executar

### Requisitos
- Compilador C++26 (`g++` 14+, `clang++` 18+ ou GCC 16)
- `cmake` (3.25+) e `ninja` ou `make`

### Compilação
```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Executar Testes Unitários, Aceitação e Classificador
```bash
ctest --test-dir build --output-on-failure
# ou individualmente:
./build/test_math
./build/test_dataset
./build/test_acceptance
./build/test_classifier
./build/test_tokenizer
./build/test_text_tokenizer
./build/test_attention
```

### Executar a API de Classificação de Produção (CLI & Benchmark)
```bash
# Executa inferências ricas sobre cenários nominais, degradados, contraditórios e benchmark de 100.000 amostras:
./build/tso_classifier
```

### Executar Experimentos
```bash
# EXP-001: Julgamento Calibrado em Microcosmo Estruturado
./build/tso_exp001

# EXP-002: Julgamento Triplo Multi-Head (Choice + Locus + Score)
./build/tso_exp002

# EXP-003: Missing Values Explícitos e Curva de Degradação sob Incerteza
./build/tso_exp003

# EXP-004: Evidência Contraditória, Resolução de Conflitos e Temperature Scaling
./build/tso_exp004

# EXP-005: Detecção de OOD por Energia Livre e Predição Seletiva
./build/tso_exp005

# EXP-006: Representações Semanticamente Equivalentes e Variedades Latentes Invariantes
./build/tso_exp006

# EXP-007: Tokenização Discreta, Embeddings Treináveis e Topologia Semântica
./build/tso_exp007

# EXP-008: Tiny Multi-Head Self-Attention, Roteamento Contextual e Heatmaps
./build/tso_exp008

# EXP-009: Linguagem Semi-Estruturada em Microdomínio e Invariância Sintática
./build/tso_exp009

# EXP-010: Julgamento Probabilístico Tipado sobre Estados Textuais em Produção (v1.0.0)
./build/tso_exp010
```

### Inferência Direta via CLI sobre Sentenças em Linguagem Natural
```bash
# Inferência textual em tempo real via CLI
./build/tso_classifier --text "declaration is true and registration is true. runtime process is running and witness is valid. freshness is fresh and health is healthy."
```

---

## Roadmap

- [x] **v0.1**: Estado estruturado $\rightarrow$ Choice calibrado + Métricas ECE/Brier/Entropia (EXP-001)
- [x] **v0.2**: Choice + Locus (atribuição diagnóstica) + Score contínuo calibrado (EXP-002)
- [x] **v0.3**: Missing values explícitos e aprendizado de incerteza (EXP-003)
- [x] **v0.4**: Evidência contraditória e robustez de calibração (EXP-004)
- [x] **v0.5**: OOD, Predição Seletiva / Abstenção Formal e Universo Canônico $\Omega=576$ (EXP-005)
- [x] **v0.6**: Representações semanticamente equivalentes e variedades latentes invariantes (EXP-006)
- [x] **v0.7**: Tokens / Embeddings Treináveis e Topologia Semântica Emergente (EXP-007)
- [x] **v0.8**: Tiny Attention Encoder (Self-Attention $QK^T/\sqrt{d_k}V$ Multi-Head & Pre-LN Block) (EXP-008)
- [x] **v0.9**: Linguagem semi-estruturada em microdomínio controlado (EXP-009)
- [x] **v1.0**: Julgamento probabilístico tipado sobre estados textuais (Local-First, Minimal Model) (EXP-010)
- [ ] **v2.0 (Horizonte de Pesquisa)**: *Question-Conditioned Typed Judgment* ($\text{State} + \text{Typed Question} \to \text{Typed Probabilistic Answer}$)


