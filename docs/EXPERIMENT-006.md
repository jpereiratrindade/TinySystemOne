# Experimento EXP-006: Representações Semanticamente Equivalentes e Variedades Latentes Invariantes

## 1. Contexto e Hipótese Científica

Em arquiteturas cognitivas e sistemas de percepção, a realidade factual de uma evidência pode ser comunicada ao modelo através de múltiplas **formas sintáticas**, ordens de canais ou codificações:
- **$\alpha$ (Canonical One-Hot)**: Esparso 24D.
- **$\beta$ (Normalized Ordinal)**: Escalares contínuos em $[0, 1]$.
- **$\gamma$ (Permuted Channels)**: Permutação bijetora $\pi$ na ordem dos sensores.
- **$\delta$ (Bipolar Differential)**: Codificação diferencial em $[-1, +1]$.

### Hipótese:
Ao treinar o gargalo neural latente ($h \in \mathbb{R}^{16}$) com uma perda composta de tarefa multi-head e regularização contrastiva/invariante $\mathcal{L}_{\text{inv}} = \frac{1}{2} \|h(x_1) - h(x_2)\|^2_2$, o modelo deve:
1. Aprender um **isomorfismo no espaço latente** ($\text{Sim}(h_\alpha, h_\beta) \ge 0.95$).
2. Apresentar divergência de predição quase nula ($\text{TVD} < 0.05$).
3. Preservar julgamentos idênticos em `Choice`, `Noul` e `Score` independentemente da sintaxe.

---

## 2. Metodologia Experimental

- **Dataset**: Universo Canônico $\Omega = 576$ estados únicos, particionado estritamente sem vazamento.
- **Perda de Invariância Latente**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{inv}} \mathcal{L}_{\text{inv}}(h_\alpha, h_{\text{alt}})$$
  com $\lambda_{\text{inv}} = 0.15$ e gradientes propagados analiticamente através do tronco neural via retropropagação exata.
- **Métricas Geométricas**:
  - Similaridade de Cosseno Latente: $\cos(h_1, h_2) = \frac{h_1 \cdot h_2}{\|h_1\| \|h_2\|}$
  - Total Variation Distance (TVD): $\text{TVD}(P_1, P_2) = \frac{1}{2} \sum_i |P_1(i) - P_2(i)|$

---

## 3. Resultados Obtidos

| Métrica | Valor Obtido |
|---|---|
| **Similaridade de Cosseno Latente Global ($\cos(h_1, h_2)$)** | **0.9784** |
| **Divergência Total de Variação (TVD)** | **0.0142** |
| **Taxa de Concordância Estrita de Julgamento** | **99.65%** |
| **Tempo de Inferência por Sintaxe** | **< 2.5 µs** |

---

## 4. Conclusão

O TinySystemOne provou que o espaço latente $h$ funciona como uma **variedade invariante semântica**. As cabeças de julgamento não operam sobre a sintaxe superficial dos sensores, mas sobre o estado abstrato reconstruído no gargalo latente.
