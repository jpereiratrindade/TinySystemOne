# TinySystemOne

> **Investigando experimentalmente julgamento probabilístico tipado, incerteza e calibração em C++26.**

O **TinySystemOne** é uma implementação white-box de uma arquitetura neural deliberada, minimalista e autônoma, projetada do zero para estudar a emergência de julgamento, confiança e incerteza sobre evidências completas, incompletas e contraditórias.

---

## Características

- 🚀 **100% C++26 puro**: Zero frameworks pesados (sem PyTorch, LibTorch, ONNX), zero dependência de rede, zero dependência de serviços externos.
- 📐 **White-Box Neural Engine**: Álgebra linear, ativadores (GELU, ReLU, Sigmoid, Softplus, Softmax), retropropagação analítica exata e otimizadores (AdamW, SGD) implementados explicitamente.
- 🎯 **Incerteza e Calibração de 1ª Classe**: Shannon Entropy, Brier Score, Expected Calibration Error (ECE) e **Temperature Scaling Post-Hoc** com otimizador analítico em C++26 puro.
- 🧠 **Resolução de Conflitos e Incerteza (v0.4.0)**:
  - **Choice**: qualidade do estado (`NOMINAL`, `DEGRADED`, `INCONSISTENT`, `UNKNOWN`).
  - **Noul**: atribuição causal do conflito / anomalia.
  - **Score**: grau contínuo de viabilidade ($\hat{\mu}$).
  - **Uncertainty**: variância epistêmica ($\hat{\sigma}^2$).

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

### Executar Testes Unitários
```bash
ctest --test-dir build --output-on-failure
# ou individualmente:
./build/test_math
./build/test_dataset
```

### Executar Experimentos
```bash
# EXP-001: Julgamento Calibrado em Microcosmo Estruturado
./build/tso_exp001

# EXP-002: Julgamento Triplo Multi-Head (Choice + Noul + Score)
./build/tso_exp002

# EXP-003: Missing Values Explícitos e Curva de Degradação sob Incerteza
./build/tso_exp003

# EXP-004: Evidência Contraditória, Resolução de Conflitos e Temperature Scaling
./build/tso_exp004
```

---

## Roadmap

- [x] **v0.1**: Estado estruturado $\rightarrow$ Choice calibrado + Métricas ECE/Brier/Entropia (EXP-001)
- [x] **v0.2**: Choice + Noul (atribuição causal) + Score contínuo calibrado (EXP-002)
- [x] **v0.3**: Missing values explícitos e aprendizado de incerteza (EXP-003)
- [x] **v0.4**: Evidência contraditória e robustez de calibração (EXP-004)
- [ ] **v0.5**: OOD e generalização
- [ ] **v0.6**: Representações semanticamente equivalentes
- [ ] **v0.7**: Tokens / Embeddings
- [ ] **v0.8**: Tiny Attention Encoder
- [ ] **v0.9**: Linguagem semi-estruturada
- [ ] **v1.0**: Julgamento probabilístico tipado sobre estados textuais
