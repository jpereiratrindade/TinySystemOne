# EXP-001: Julgamento Calibrado em Microcosmo Estruturado

## Pergunta Experimental

> **Um modelo pequeno consegue distinguir:**
> 1. Estado conhecido e consistente (`NOMINAL`);
> 2. Estado conhecido e degradado (`DEGRADED`);
> 3. Evidência contraditória (`INCONSISTENT`);
> 4. Informação insuficiente / ausente (`UNKNOWN`);
>
> **sem apresentar confiança elevada indiscriminadamente sob ambiguidade ou dados fora de distribuição (OOD)?**

---

## Dimensões do Estado (Vetor de 18 dimensões)

| Campo | Categorias / Valores | Dims |
| :--- | :--- | :---: |
| `declared` | `[true, false]` | 2 |
| `registered` | `[true, false]` | 2 |
| `runtime` | `[running, absent, unknown]` | 3 |
| `witness` | `[valid, invalid, stale, unknown]` | 4 |
| `freshness` | `[fresh, aging, expired]` | 3 |
| `health` | `[healthy, degraded, failing, unknown]` | 4 |
| **Total** | | **18** |

---

## Classes de Saída (`Choice`)

- `0: NOMINAL`: Estado coerente, ativo e saudável.
- `1: DEGRADED`: Ativo porém com saúde degradada, dados expirados ou falha declarada.
- `2: INCONSISTENT`: Conflito factual explícito (ex: runtime ativo mas witness inválido, ou registrado sem declaração).
- `3: UNKNOWN`: Informações essenciais omitidas ou estado indeterminado (`unknown` dominante).

---

## Protocolo de Avaliação

1. **Partição de Dados**:
   - Treino: 80% dos exemplos sintéticos regulares.
   - Validação: 10%.
   - Teste: 10%.
   - **OOD / Incerteza Extrema**: Conjunto separado com vetores de ruído, conflitos bizarros ou ativações parciais não vistas no treino.
2. **Critérios de Sucesso**:
   - Acurácia no conjunto de teste $\ge 95\%$.
   - Erro de Calibração Esperado (ECE) $\le 0.08$.
   - Entropia de Shannon no conjunto OOD significativamente maior que no conjunto NOMINAL/DEGRADED claro (o modelo expressa dúvida quando confrontado com o desconhecido).
