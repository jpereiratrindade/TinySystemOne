# TinySystemOne — Architecture (v0.1.0)

## Overview

O TinySystemOne v0.1.0 adota uma arquitetura modular, determinística e sem dependências externas em **C++26**:

```text
       ┌────────────────────────┐
       │ Structured State (JSON/│
       │ Struct)                │
       └───────────┬────────────┘
                   │
                   ▼
         [ One-Hot Encoding ]  →  Input Vector x ∈ R^18
                   │
                   ▼
       ┌────────────────────────┐
       │   Linear Layer 18 → 32 │
       │   GELU / ReLU          │
       │   Linear Layer 32 → 16 │
       │   GELU / ReLU          │
       │   Linear Layer 16 → 4  │
       └───────────┬────────────┘
                   │
                   ▼
         [ Softmax Layer ]     →  Probability Distribution P ∈ [0,1]^4
                   │
       ┌───────────┴────────────────────────┐
       │ Choice Head:                       │
       │  0: NOMINAL                        │
       │  1: DEGRADED                       │
       │  2: INCONSISTENT                   │
       │  3: UNKNOWN                        │
       └────────────────────────────────────┘
                   │
                   ▼
       ┌────────────────────────────────────┐
       │ Calibration & Metrics:             │
       │  - Top-1 Choice & Confidence       │
       │  - Shannon Entropy H(P)            │
       │  - Brier Score                     │
       │  - Expected Calibration Error (ECE)│
       └────────────────────────────────────┘
```

## Componentes

### 1. `tso::Tensor` e Álgebra (`include/tso/tensor.hpp`)
- `Vector`: representação contígua `std::vector<float>` de 1D com operações element-wise, dot product e normas.
- `Matrix`: armazenamento contíguo `std::vector<float>` com layout row-major `(rows, cols)`.
- Funções de ativação vetorizadas:
  - `ReLU(x) = max(0, x)`, `dReLU(x) = (x > 0 ? 1 : 0)`
  - `GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
  - `Softmax(z)_i = exp(z_i - max(z)) / sum_j exp(z_j - max(z))`

### 2. `tso::Random` (`include/tso/random.hpp`)
- Motor pseudo-aleatório `std::mt19937_64` reproduzível por semente.
- Inicializadores de pesos:
  - **He (Kaiming) Normal**: $\mathcal{N}(0, \sqrt{2 / d_{in}})$ para camadas com ReLU/GELU.
  - **Xavier (Glorot) Uniform**: $\mathcal{U}(-\sqrt{6/(d_{in}+d_{out})}, \sqrt{6/(d_{in}+d_{out})})$.

### 3. `tso::Model` (`include/tso/model.hpp`)
- Rede MLP multicamada com estado de ativação retido durante o `forward()` para cálculo de gradiente exato no `backward()`.
- Cálculo analítico de derivadas $\frac{\partial L}{\partial W}$ e $\frac{\partial L}{\partial b}$.

### 4. `tso::Optimizer` (`include/tso/optimizer.hpp`)
- **AdamW**: com momentos $m_t$, $v_t$, correção de bias ($\hat{m}_t, \hat{v}_t$) e weight decay desacoplado ($\lambda$).
- **SGD**: com momentum de Polyak.

### 5. `tso::Calibration` (`include/tso/calibration.hpp`)
- Métricas formais para quantificar se a confiança condiz com a probabilidade real de acerto.
- ECE particionado em $M$ bins equiprováveis / uniformes no intervalo $[0, 1]$.
