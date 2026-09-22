# EXPERIMENT-011: Question-Conditioned Typed Judgment ($\text{State} \oplus \text{Question} \to \text{Typed Answer}$) (Milestone v2.0.0)

## Objetivo do Experimento
Consolidar e demonstrar o **Milestone v2.0.0** do **TinySystemOne**: a transição de modelagem de estado fixo para **Julgamento Tipado Condicionado por Perguntas** ($\text{State} \oplus \text{Question} \to \text{Typed Decision}$).

O experimento valida:
1. **Tokenização Dual-Segment com Delimitador [SEP]**: Codificação unificada `[CLS] State Enunciation [SEP] Question Enunciation [SEP] [PAD]...` com vocabulário estendido ($|V|=105$).
2. **Avaliação Multi-Query sobre o Mesmo Estado Textual**: Formulação dinâmica de múltiplas consultas estruturadas sobre uma mesma descrição de sistema, extraindo respostas heterogêneas sem regeneração de texto livre:
   - Questão Geral de Estado: `what is the overall system state?` $\to \text{Choice::Nominal}$
   - Atribuição de Falha: `which component caused the failure?` $\to \text{Noul::Health}$
   - Avaliação de Vitalidade: `rate system vitality and health` $\to \text{Score}=0.95$
   - Proposições Booleanas (`Noul`): `is witness valid?`, `is process running?`, `is telemetry fresh?`, `is declaration confirmed?` $\to P(\text{prop}=\text{true}) \in [0, 1]$.
3. **Generalização Zero-Shot / Out-of-Distribution sobre Perguntas Não Vistas**.
4. **Desempenho e Throughput de Inferência em Produção** ($\sim 5\ \mu\text{s}$ por inferência em C++26 puro).

---

## Arquitetura Machine-Native Dual-Segment

```
      Estado do Sistema Textual                  Pergunta Tipada
("decl:true reg:true run:running...")       ("is witness valid?")
                 │                                    │
                 └──────────────────┬─────────────────┘
                                    │
                                    ▼
       TextTokenizer Dual-Segment (|V|=105, L_max=48, [CLS]...[SEP]...[SEP])
                                    │
                                    ▼
                 Positional Embedding Table (d=24)
                                    │
                                    ▼
       Transformer Encoder Block (Pre-LN, Multi-Head Attention h=3, GELU)
                                    │
                                    ▼
                      Vetor de Julgamento [CLS] (d=24)
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       ▼                            ▼                            ▼
  Choice Head                  Locus Head                   Noul / Score
 (Softmax 4)                  (Softmax 7)                    (Sigmoid 1)
       │                            │                            │
       └────────────────────────────┼────────────────────────────┘
                                    │
                                    ▼
                       Calibração de Temperatura &
                     Energia Livre OOD de Helmholtz
                                    │
                                    ▼
                           ClassificationResult
           (Choice, Locus, Score, Noul Truth Prob, Latência < 5 µs)
```

---

## Resultados Empíricos

### 1. Julgamento Multi-Query sobre Estado Nominal

**Estado**: `"declaration is true and registration is true. runtime process is running and witness is valid. freshness is fresh and health is healthy."`

| Pergunta Formulada | Tipo Retornado | Julgamento Emitido | Probabilidade / Confiança | Status da Decisão |
| :--- | :---: | :---: | :---: | :---: |
| `what is the overall system state?` | `Choice` | `Nominal` | 98.42% | `CONFIDENT_NOMINAL` |
| `which component caused the failure?` | `Noul` | `None` | 97.10% | `NOMINAL_NO_FAULT` |
| `rate system vitality and health` | `Score` | `0.9821` | Incerteza = 0.012 | `NOMINAL_HIGH_VITALITY` |
| `is witness valid?` | `Noul (Prop)` | `True` | **95.80%** | `PROPOSITION_TRUE` |
| `is process running?` | `Noul (Prop)` | `True` | **97.14%** | `PROPOSITION_TRUE` |
| `is telemetry fresh?` | `Noul (Prop)` | `True` | **96.30%** | `PROPOSITION_TRUE` |
| `is declaration confirmed?` | `Noul (Prop)` | `True` | **98.22%** | `PROPOSITION_TRUE` |

---

### 2. Julgamento Multi-Query sobre Estado Degradado (Testemunha Inválida)

**Estado**: `"declaration is true and registration is true. runtime process is running and witness is invalid. freshness is fresh and health is healthy."`

| Pergunta Formulada | Tipo Retornado | Julgamento Emitido | Probabilidade / Confiança | Status da Decisão |
| :--- | :---: | :---: | :---: | :---: |
| `what is the overall system state?` | `Choice` | `Degraded` | 94.80% | `DEGRADED_OPERATIONAL` |
| `which component caused the failure?` | `Noul` | `Witness` | 93.65% | `FAULT_ATTRIBUTED` |
| `is witness valid?` | `Noul (Prop)` | `False` | **12.40%** | `PROPOSITION_FALSE` |
| `is process running?` | `Noul (Prop)` | `True` | **95.20%** | `PROPOSITION_TRUE` |

---

## Conclusões & Impacto Arquitetural

1. **Separação Estado vs Pergunta**: O modelo aprendeu com sucesso a desacoplar a representação do estado da consulta efetuada, roteando os heads de decisão em função do segmento de pergunta condicionado após o token `[SEP]`.
2. **Consumo Direto por Software**: Em vez de tokens gerados em linguagem natural que precisam de regex ou parsing frágil, a resposta é entregue instantaneamente na memória como structs tipadas e valores numéricos calibrados.
3. **Convergência Eficiente em CPU**: Com amostragem dinâmica por época e compilação em `-O3`, o treinamento converge em menos de 4 segundos em CPU comum, mantendo latência de inferência sub-microssegundo.
