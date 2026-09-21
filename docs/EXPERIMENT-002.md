# EXP-002: Julgamento Triplo Simultâneo (Choice + Noul + Score)

## Pergunta Experimental

> **Um modelo pequeno com arquitetura Multi-Head e representação latente compartilhada consegue aprender simultaneamente:**
> 1. **Qualidade do estado** (`Choice`: `NOMINAL`, `DEGRADED`, `INCONSISTENT`, `UNKNOWN`);
> 2. **Atribuição causal do julgamento** (`Noul`: `NONE`, `DECLARATION`, `RUNTIME`, `WITNESS`, `FRESHNESS`, `HEALTH`, `MULTIPLE`);
> 3. **Grau contínuo de viabilidade operacional** (`Score` $\in [0.0, 1.0]$);
>
> **mantendo calibração probabilística em ambas as saídas categóricas e convergência suave na regressão contínua?**

---

## Arquitetura Multi-Head

```text
               Input Vector x ∈ R^18
                         │
                         ▼
        ┌──────────────────────────────────┐
        │ Trunk Compartilhado              │
        │   Linear 18 → 32 (GELU)          │
        │   Linear 32 → 16 (GELU)          │
        └────────────────┬─────────────────┘
                         │ Representação Latente h ∈ R^16
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
   ┌───────────┐   ┌───────────┐   ┌───────────┐
   │Choice Head│   │ Noul Head │   │Score Head │
   │  16 → 4   │   │  16 → 7   │   │  16 → 1   │
   │  Softmax  │   │  Softmax  │   │  Sigmoid  │
   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
         │               │               │
         ▼               ▼               ▼
      Choice           Noul            Score
   Probabilities   Probabilities    Scalar [0,1]
```

---

## Função de Perda Conjunta

$$\mathcal{L}_{total} = \mathcal{L}_{choice}(\hat{c}, c) + \mathcal{L}_{noul}(\hat{n}, n) + 2.0 \cdot \mathcal{L}_{MSE}(\hat{s}, s)$$

Os gradientes das três cabeças convergem no gargalo latente $h$:

$$\frac{\partial \mathcal{L}_{total}}{\partial h} = \frac{\partial \mathcal{L}_{choice}}{\partial h} + \frac{\partial \mathcal{L}_{noul}}{\partial h} + \frac{\partial \mathcal{L}_{score}}{\partial h}$$

---

## Critérios de Sucesso

1. **Choice Accuracy** $\ge 98\%$, ECE $\le 0.05$.
2. **Noul Accuracy** $\ge 95\%$, ECE $\le 0.05$.
3. **Score MSE** $\le 0.005$ (erro médio quadrático reduzido) com correlação alta com o ground truth.
4. Sob **OOD**, o modelo reduz a confiança em Choice/Noul e atribui Scores degradados compatíveis com a incerteza.
