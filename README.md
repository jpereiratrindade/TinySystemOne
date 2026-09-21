# TinySystemOne

> **Investigando experimentalmente julgamento probabilístico tipado, incerteza e calibração em C++26.**

O **TinySystemOne** é uma implementação white-box de uma arquitetura neural deliberada, minimalista e autônoma, projetada do zero para estudar a emergência de julgamento, confiança e incerteza sobre evidências completas, incompletas e contraditórias.

---

## Características

- 🚀 **100% C++26 puro**: Zero frameworks pesados (sem PyTorch, LibTorch, ONNX), zero dependência de rede, zero dependência de serviços externos.
- 📐 **White-Box Neural Engine**: Álgebra linear, ativadores (GELU, ReLU, Sigmoid, Softmax), retropropagação analítica exata e otimizadores (AdamW, SGD) implementados explicitamente.
- 🎯 **Incerteza e Calibração de 1ª Classe**: Shannon Entropy, Brier Score e Expected Calibration Error (ECE) integrados no pipeline.
- 🧠 **Arquitetura Multi-Head (v0.2.0)**: Predição conjunta de **Choice** (qualidade), **Noul** (atribuição causal) e **Score** (grau escalar contínuo calibrado).

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
```

---

## Roadmap

- [x] **v0.1**: Estado estruturado $\rightarrow$ Choice calibrado + Métricas ECE/Brier/Entropia (EXP-001)
- [x] **v0.2**: Choice + Noul (atribuição causal) + Score contínuo calibrado (EXP-002)
- [ ] **v0.3**: Missing values explícitos e aprendizado de incerteza
- [ ] **v0.4**: Evidência contraditória e robustez de calibração
- [ ] **v0.5**: OOD e generalização
- [ ] **v0.6**: Representações semanticamente equivalentes
- [ ] **v0.7**: Tokens / Embeddings
- [ ] **v0.8**: Tiny Attention Encoder
- [ ] **v0.9**: Linguagem semi-estruturada
- [ ] **v1.0**: Julgamento probabilístico tipado sobre estados textuais
