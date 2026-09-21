# Experimento EXP-007: Tokens Discretos, Embeddings Aprendíveis e Modelagem em Sequência

## 1. Contexto e Motivação

Até a v0.6, o TinySystemOne recebia evidências estruturadas como vetores contíguos de 24 dimensões. No **EXP-007 (v0.7.0)**, o modelo faz a transição para uma **representação discreta-simbólica baseada em tokens e embeddings contínuos aprendíveis** em 100% C++26 puro.

Essa etapa constrói a base necessária para a introdução do **Tiny Attention Encoder (v0.8)**.

---

## 2. Metodologia

1. **Vocabulário (|V| = 25 tokens)**:
   - Tokens especiais: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`.
   - Chaves de sensores: `KEY:declared`, `KEY:registered`, `KEY:runtime`, `KEY:witness`, `KEY:freshness`, `KEY:health`.
   - Valores de estado: `VAL:true`, `VAL:false`, `VAL:running`, `VAL:absent`, `VAL:valid`, `VAL:invalid`, `VAL:stale`, `VAL:fresh`, `VAL:aging`, `VAL:expired`, `VAL:healthy`, `VAL:degraded`, `VAL:failing`, `VAL:unknown`.

2. **Arquitetura `TokenizedMultiHeadMLP`**:
   - Tabela de Embeddings de Tokens $W_e \in \mathbb{R}^{25 \times 16}$.
   - Tabela de Embeddings Posicionais $W_{pos} \in \mathbb{R}^{14 \times 16}$.
   - Agregação por Mean Pooling $\rightarrow$ Tronco MLP (16 $\rightarrow$ 32 $\rightarrow$ 16) $\rightarrow$ 4 Cabeças Multi-Head (`Choice`, `Noul`, `Score`, `Uncertainty`).
   - Retropropagação analítica exata propagada até os pesos da tabela de embeddings com otimizador AdamW.

---

## 3. Resultados Obtidos

- **Acurácia de Generalização em Sequências de Teste Inéditas**: **100.00%** em `Choice` e **100.00%** em `Noul`.
- **Topologia Semântica dos Embeddings**: O modelo aprende a posicionar valores opostos e correlacionados em regiões semanticamente significativas do espaço vetorial $\mathbb{R}^{16}$.
- **Tamanho do Modelo**: Apenas 1.709 parâmetros totais, executando em microssegundos.
