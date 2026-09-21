# EXPERIMENT-009: Semi-Structured Language in Controlled Microdomain

## Objetivo do Experimento
Avaliar a capacidade de generalização e invariância sintática do **TinySystemOne (v0.9.0)** ao substituir sequências puramente simbólicas por **enunciados textuais em linguagem semi-estruturada**, relatórios de incidentes, logs de telemetria e sentenças em prosa natural dentro de um microdomínio controlado.

O foco central é responder empiricamente:
> **Um modelo white-box em C++26 puro (~10k parâmetros) consegue produzir julgamentos tipados, calibrados e com atribuição diagnóstica a partir de texto livre sem gerar texto?**

---

## Metodologia & Arquitetura

### 1. Tokenizador Lexical de Microdomínio ([text_tokenizer.hpp](file:///home/jpereiratrindade/dev/cpp/TinySystemOne/include/tso/text_tokenizer.hpp))
- **Vocabulário**: $|V| = 83$ tokens cobrindo pontuação, substantivos de domínio (`service`, `declaration`, `runtime`, `witness`, `freshness`, `health`), adjetivos (`true`, `false`, `running`, `absent`, `valid`, `stale`, `healthy`, `degraded`, `failing`), conectivos (`is`, `and`, `with`, `at`) e tokens especiais (`[CLS]`, `[SEP]`, `[PAD]`, `[UNK]`).
- **Sequenciamento**: Comprimento fixo $L_{max} = 32$ tokens com padding explícito e delimitadores `[CLS]` e `[SEP]`.

### 2. Multi-Estilo Sintático de Estados ([dataset.hpp](file:///home/jpereiratrindade/dev/cpp/TinySystemOne/include/tso/dataset.hpp))
Para cada estado no Universo Canônico $\Omega=576$, geram-se 4 famílias de apresentação textual:
1. **Prosa Natural**: `"declaration is true and registration is true. runtime process is running and witness is valid. freshness is fresh and health is healthy."`
2. **Log de Telemetria**: `"[status] decl=true | reg=true | run=running | wit=valid | fresh=fresh | hlth=healthy"`
3. **Relatório Diagnóstico de Incidente**: `"service declaration: true; discovery: true; execution: running; verification: valid; telemetry: fresh; metric: healthy"`
4. **Chave-Valor Compacto**: `"decl:true reg:true run:running wit:valid fresh:fresh hlth:healthy"`

### 3. Modelo Neural Textual
- $d_{model} = 24$, $h = 3$ cabeças de atenção ($d_k = 8$), $d_{ff} = 48$, GELU.
- Total de parâmetros: **10,333** (Tabelas de Embeddings: 2,760, Transformer Pre-LN: 4,872, Heads de Decisão: 2,701).

---

## Resultados Empíricos (Zero-Shot Generalization no Test Split $\Omega=576$)

| Família Sintática Textual | Choice Accuracy (Zero-Shot) | Locus Accuracy (Zero-Shot) | Score Contínuo (MSE) | Expected Calibration Error (ECE) | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Prosa Natural** | **86.25%** | **77.50%** | 0.0098 | 0.0777 | 0.2394 |
| **2. Log de Telemetria** | **85.00%** | **76.25%** | 0.0104 | 0.1129 | 0.2502 |
| **3. Relatório Diagnóstico** | **85.00%** | **77.50%** | 0.0108 | 0.0764 | 0.2467 |
| **4. Chave-Valor Compacto** | **85.00%** | **75.00%** | 0.0130 | 0.1282 | 0.2765 |

---

## Descobertas Científicas & Análise

1. **Invariância Sintática Estável**: O modelo mantém uma acurácia de escolha de ~85% em todas as 4 variações estilísticas em estados nunca vistos durante o treino. A variação de acurácia entre prosa natural e log delimitado por pipe é de apenas $\pm 1.25\%$.
2. **Emergência de Julgamento Textual sem Geração de Texto**: O vetor `[CLS]` contextualizado sintetiza o texto em representação densa que permite aos heads clássicos (`Choice`, `Locus`, `Score`) julgar a coerência da observação diretamente.
3. **Escalabilidade com Minimal Sufficient Model**: O número de parâmetros subiu de ~4k para ~10k para acomodar o vocabulário textual e sequências maiores ($L=32$), permanecendo executável em tempo de sub-microssegundo e 100% offline.
